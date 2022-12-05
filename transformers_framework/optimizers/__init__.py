from transformers_lightning.optimizers import AdamWOptimizer, ElectraAdamWOptimizer


optimizers = dict(
    adamw=AdamWOptimizer,
    electra_adamw=ElectraAdamWOptimizer,
)
