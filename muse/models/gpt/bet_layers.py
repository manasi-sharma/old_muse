"""
GPT model FROM BET:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier

Taken from: https://github.com/notmahi/bet/blob/main/models/libraries/mingpt/model.py
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from muse.experiments import logger


class TransformerConfig:
    """base Transformer config, params common to all GPT versions"""

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    discrete_input = False
    input_size = 10
    n_embd = 768
    n_layer = 12
    n_head = None
    causal = True
    causal_block_size = 1
    use_temporal_embd = True

    def __init__(self, vocab_size, block_size, **kwargs):
        # input / output size (just output if !discrete_input)
        self.vocab_size = vocab_size
        # horizon length
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(TransformerConfig):
    """GPT-1 like network roughly 125M params"""

    n_layer = 12
    n_head = 12
    n_embd = 768


class SelfAttention(nn.Module):
    """
        A vanilla multi-head masked self-attention layer with a projection at the end.
        It is possible to use torch.nn.MultiheadAttention here but I am including an
        explicit implementation here to show that there is nothing too scary here.
        """

    def __init__(self, config):
        super().__init__()
        # assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_head * config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_head * config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_head * config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_head * config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.n_head = config.n_head

    def forward(self, x, mask=None):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(B, T, self.n_head, C).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C).transpose(1, 2)
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if mask is not None:
            att = att.masked_fill(mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, self.n_head*C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class CausalSelfAttention(SelfAttention):
    def __init__(self, config):
        super().__init__(config)
        if config.causal_block_size > 1:
            assert config.block_size % config.causal_block_size == 0,f"Mask block size {config.causal_block_size} must evenly divide block size {config.block_size}"
            h_size = config.block_size // config.causal_block_size
            mask = torch.tril(torch.ones(h_size, h_size)).view(
                    1, 1, h_size, h_size
                )
            mask = mask.repeat_interleave(config.causal_block_size, dim=-1).repeat_interleave(config.causal_block_size, dim=-2)
        else:
            mask = torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                )
        self.register_buffer(
            "mask", mask
        )
    
    def forward(self, x, **kwargs):
        mask = self.mask
        if 'mask' in kwargs.keys() and kwargs['mask'] is not None:
            # only attend where both mask and causal mask are 1.
            mask = mask * kwargs['mask']
        return super().forward(x, mask=mask)


class TransformerBlock(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config) if config.causal else SelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        self.mask = None

    def set_mask(self, mask):
        # will be combined with optional casual mask, cleared after one call to forward.
        self.mask = mask

    def forward(self, x):
        x = x + self.attn(self.ln1(x), mask=self.mask)
        x = x + self.mlp(self.ln2(x))
        self.mask = None
        return x


class GPT(nn.Module):
    """the full GPT language model, with a context size of block_size"""

    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.cfg = config

        # input embedding stem
        if config.discrete_input:
            self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        else:
            self.tok_emb = nn.Linear(config.input_size, config.n_embd)
        self.discrete_input = config.discrete_input
        if config.use_temporal_embd:
            self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e" % sum(p.numel() for p in self.parameters())
        )

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, GPT):
            if module.cfg.use_temporal_embd:
                torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)

    def configure_optimizers(self, train_config):
        """
        TODO incorporate this

        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        if self.cfg.use_temporal_embd:
            # special case the position embedding parameter in the root GPT module as not decayed
            no_decay.add("pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer

    def set_mask(self, mask):
        for block in self.blocks:
            block.set_mask(mask)

    def forward(self, idx):
        """
        Embeds input idxs (discrete or continuous) into output logits (or latents)
        """
        
        if self.discrete_input:
            b, t = idx.size()
        else:
            b, t, dim = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
        if self.cfg.use_temporal_embd:
            position_embeddings = self.pos_emb[
                :, :t, :
            ]  # each position maps to a (learnable) vector
            token_embeddings = token_embeddings + position_embeddings
        x = self.drop(token_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        latent = self.head(x)

        return latent
