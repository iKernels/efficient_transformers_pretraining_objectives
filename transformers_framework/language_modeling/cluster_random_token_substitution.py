from typing import Tuple

import torch
from transformers import PreTrainedTokenizerBase
from transformers_lightning.language_modeling import IGNORE_IDX, LanguageModel


class ClusterRandomTokenSubstitution(LanguageModel):
    r"""
    Prepare tokens inputs/labels for spanned replaced token detection modeling.
    We sample a few spans in each sequence for RTS training (with probability `probability`
    defaults to 0.15 in Bert/RoBERTa).

    `n_clusters` and `token_to_cluster` should be provided if you want the replacements to be chosen with a criteria
    based on past predictions.

    Example:
        >>> import torch
        >>> from transformers import BertTokenizer

        >>> tok = BertTokenizer.from_pretrained("bert-base-cased")
        >>> rts = RandomTokenSubstitution(tok)

        >>> input_ids = torch.tensor([tok.encode("what can transformers do?")])
        >>> # tokens: ['[CLS]', 'what', 'can', 'transform', '##ers', 'do', '?', '[SEP]']

        >>> input_ids
        tensor([[101, 1184, 1169, 11303, 1468, 1202, 136, 102]])

        >>> replaced, labels = rts(input_ids)
        >>> replaced
        tensor([[ 101, 2774, 5650, 102]])
        >>> labels
        tensor([[-100, 1, 0, -100]]) # -100 = IGNORE_IDX
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        probability: float = 0.15,
        model: torch.nn.Module = None,
        beta: float = 2.0,
    ):
        super().__init__(tokenizer, probability=probability)

        assert model is not None and hasattr(model, 'token_to_cluster_map') and hasattr(model, 'counts'), (
            "Model must be not None and have both `counts` and `token_to_cluster_map` attributes"
        )
        self.model = model
        self.beta = beta
        self.candidates = None

    def __call__(self, inputs: torch.Tensor) -> Tuple[torch.LongTensor, torch.LongTensor]:
        device = inputs.device
        inputs = inputs.clone()
        labels = torch.full(inputs.shape, fill_value=0, dtype=torch.long, device=device)

        # We sample a few tokens in each sequence for random replaced language modeling
        # training (with probability self.probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(inputs.shape, fill_value=self.probability, dtype=torch.float32, device=device)

        # not going to substitute special tokens of the LM (bert, roby, ...)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()
        ]
        special_tokens_mask_tensor = torch.tensor(special_tokens_mask, dtype=torch.bool, device=device)
        probability_matrix.masked_fill_(special_tokens_mask_tensor, value=0.0)
        labels.masked_fill_(special_tokens_mask_tensor, value=IGNORE_IDX)

        # no need to substitute padding tokens, assigning 0.0 prob
        if self.tokenizer._pad_token is not None:
            padding_mask = inputs.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
            labels.masked_fill_(padding_mask, value=IGNORE_IDX)

        # sample substitutions from probability distribution
        substituted_indices = torch.bernoulli(probability_matrix).bool()

        # define a uniform probability over the elements in each cluster
        if self.candidates is None:
            candidates = self.model.token_to_cluster_map.unsqueeze(0) == torch.arange(
                self.model.counts.shape[0], device=self.model.device
            ).unsqueeze(-1)
            self.candidates = candidates / candidates.sum(-1, keepdim=True)

        # tokens_to_swap has shape (number_of_substituted_tokens,)
        tokens_to_swap = inputs[substituted_indices]

        # tokens_clusters has shape (number_of_substituted_tokens,) and contains the id of the corresponding cluster
        source_clusters = self.model.token_to_cluster_map[tokens_to_swap]

        # source_clusters has shape (number_of_substituted_tokens, n_clusters)
        target_clusters_counts = self.model.counts[source_clusters]  # 3.99 it/s

        # target_clusters_counts has shape (number_of_substituted_tokens, n_clusters)
        minimum = target_clusters_counts.min(dim=-1, keepdim=True)[0]    # take only values, not indexes
        maximum = target_clusters_counts.max(dim=-1, keepdim=True)[0]    # take only values, not indexes
        denominator = (maximum - minimum)
        denominator[denominator == 0] = 1
        target_clusters_counts_norm = (target_clusters_counts - minimum) / denominator  # 2.53 it/s

        # should do softmax - (number_of_substituted_tokens, n_clusters)
        # target_clusters_counts_norm = target_clusters_counts / torch.linalg.norm(
        #   target_clusters_counts, dim=-1, keepdim=True
        # ) # 3.06 it/s
        target_clusters_probs = torch.softmax(target_clusters_counts_norm * self.beta, dim=-1)  # 2.51 it/s

        # sample target clusters based on probabilities - (number_of_substituted_tokens,)
        sample_target_clusters = torch.multinomial(target_clusters_probs, num_samples=1).flatten()  # 1.60it/s

        # select target clusters
        target_clusters = self.candidates[sample_target_clusters]  # 1.58 it/s

        # choose words randomly from new clusters (number_of_substituted_tokens, n_clusters)
        random_words = torch.multinomial(target_clusters, num_samples=1).flatten()  # 1.56 it/s

        # substitute
        inputs[substituted_indices] = random_words
        labels.masked_fill_(substituted_indices, value=1)

        return inputs, labels
