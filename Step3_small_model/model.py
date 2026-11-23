"""Tiny GPT-style causal transformer plus helpers shared by the training
and sampling scripts."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """One transformer block: causal self-attention followed by an MLP."""

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.n_head = n_head
        self.dropout = dropout
        self.ln1 = nn.LayerNorm(n_embd)
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        B, T, C = x.shape
        hs = C // self.n_head
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        # is_causal=True: each position can only look at earlier positions.
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(y)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """A very small next-token predictor. Weights start random."""

    def __init__(self, vocab_size, block_size=64, n_layer=2, n_head=4,
                 n_embd=128, dropout=0.0):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, block_size=block_size,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                           dropout=dropout)
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate_stream(self, idx, max_new_tokens, temperature=1.0,
                        generator=None):
        """Yield one sampled internal token id at a time."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
            yield nxt.item()


def missing_tokens(enc, stoi, text):
    """Return the o200k token ids of `text` and those not in the vocabulary."""
    ids = enc.encode_ordinary(text)
    return ids, [i for i in ids if i not in stoi]


def generate_text(model, enc, stoi, itos, prompt, seed, max_new_tokens=40,
                  temperature=0.8, stream=True):
    """Generate a continuation of `prompt`, printing tokens as they appear.

    Prompt tokens are o200k_base ids; they are mapped to the compact internal
    vocabulary for the model, and generated internal ids are mapped back to
    o200k_base ids for decoding.
    """
    ids, missing = missing_tokens(enc, stoi, prompt)
    if missing:
        raise ValueError(
            f"Prompt contains tokens not in the model's vocabulary: {missing}")
    generator = torch.Generator().manual_seed(seed)
    was_training = model.training
    model.eval()
    x = torch.tensor([[stoi[i] for i in ids]], dtype=torch.long)
    out_ids = list(ids)
    shown = enc.decode(out_ids)
    if stream:
        print(shown, end="", flush=True)
    for internal_id in model.generate_stream(x, max_new_tokens, temperature,
                                             generator):
        out_ids.append(itos[internal_id])
        text = enc.decode(out_ids)
        if stream:
            print(text[len(shown):], end="", flush=True)
        shown = text
    if stream:
        print(flush=True)
    if was_training:
        model.train()
    return shown


def load_checkpoint(path):
    """Load a checkpoint saved by train.py. Returns (model, stoi, itos, ckpt)."""
    ckpt = torch.load(path, map_location="cpu")
    model = TinyGPT(**ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    itos = ckpt["itos"]
    stoi = {tok: i for i, tok in enumerate(itos)}
    return model, stoi, itos, ckpt
