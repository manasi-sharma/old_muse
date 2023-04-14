import torch
from attrdict import AttrDict as d

from muse.trainers.optimizers.tr_optimizer import TransformerSingleOptimizer

export = d(
    cls=TransformerSingleOptimizer,
    max_grad_norm=None,
    base_optimizer=d(
        cls=torch.optim.AdamW,
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    ),
),
