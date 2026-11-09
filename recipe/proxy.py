"""Small from-scratch proxy LMs for recipe experiments.

H1-style blocks (RMSNorm, RoPE, SwiGLU, GQA, tied embeddings) at a size a laptop GPU trains in about 20 minutes,
on packed uint16 token shards (any tokenizer), with synthetic biographies mixed in so fact storage can be measured
next to held-out loss. H1 is the author's own 298M model; E02-E04 used its tokenizer and training text.

    train_and_probe(recipe: dict, seed: int, out_dir) -> dict   (writes result.json, never weights)

Data comes from RECIPE_TEXT_ROOT/<source>/<split>/manifest.json (see tools/pack_text.py) and the tokenizer from
RECIPE_TOKENIZER (a Hugging Face tokenizers JSON file). RECIPE_EOS is the end-of-document id (2 in E02-E04).
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import facts as factlib

VAL_ROOT = Path(os.environ.get("RECIPE_TEXT_ROOT", "data/text"))
TOKENIZER = Path(os.environ.get("RECIPE_TOKENIZER", "data/tokenizer.json"))
EOS = int(os.environ.get("RECIPE_EOS", "2"))
BLOCK = 4096


# --------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d: int, heads: int, kv: int, ffn: int):
        super().__init__()
        self.h, self.kv, self.hd = heads, kv, d // heads
        self.n1, self.n2 = nn.RMSNorm(d, eps=1e-5), nn.RMSNorm(d, eps=1e-5)
        self.q = nn.Linear(d, heads * self.hd, bias=False)
        self.k = nn.Linear(d, kv * self.hd, bias=False)
        self.v = nn.Linear(d, kv * self.hd, bias=False)
        self.o = nn.Linear(heads * self.hd, d, bias=False)
        self.gate = nn.Linear(d, ffn, bias=False)
        self.up = nn.Linear(d, ffn, bias=False)
        self.down = nn.Linear(ffn, d, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        h = self.n1(x)
        q = self.q(h).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.k(h).view(B, T, self.kv, self.hd).transpose(1, 2)
        v = self.v(h).view(B, T, self.kv, self.hd).transpose(1, 2)
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        if self.kv != self.h:
            k = k.repeat_interleave(self.h // self.kv, 1)
            v = v.repeat_interleave(self.h // self.kv, 1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, -1))
        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


def _rope(x, cos, sin):
    hd = x.shape[-1]
    a, b = x[..., :hd // 2], x[..., hd // 2:]
    return x * cos + torch.cat([-b, a], -1) * sin


class ProxyLM(nn.Module):
    """emb_rank (E04): a word table of width r < d, widened to d by one r x d matrix `emb_proj` that is tied both
    ways (in: e @ P, out: (h @ P^T) @ E^T). Unset (or r == d) gives the original model, parameter for parameter."""

    def __init__(self, vocab: int, d: int, layers: int, heads: int, kv: int, ffn: int, theta: float = 10000.0,
                 emb_rank: int | None = None):
        super().__init__()
        r = emb_rank or d
        if not 0 < r <= d:
            raise ValueError(f"emb_rank {r} must be in 1..{d}")
        self.emb = nn.Embedding(vocab, r)
        self.blocks = nn.ModuleList([Block(d, heads, kv, ffn) for _ in range(layers)])
        self.norm = nn.RMSNorm(d, eps=1e-5)
        self.hd, self.theta = d // heads, theta
        nn.init.normal_(self.emb.weight, std=0.02)
        for blk in self.blocks:
            for m in (blk.q, blk.k, blk.v, blk.gate, blk.up):
                nn.init.normal_(m.weight, std=0.02)
            for m in (blk.o, blk.down):
                nn.init.normal_(m.weight, std=0.02 / math.sqrt(2 * layers))
        self.emb_proj = None
        if r != d:
            # orthonormal rows scaled by sqrt(d/r): the input per-dimension scale and the initial logit scale
            # match the full-width tied table, so only the table's width changes
            self.emb_proj = nn.Parameter(torch.empty(r, d))
            nn.init.orthogonal_(self.emb_proj, gain=math.sqrt(d / r))
        self._rope_cache: dict = {}

    def rope(self, T, device):
        key = (T, str(device))
        if key not in self._rope_cache:
            inv = 1.0 / (self.theta ** (torch.arange(0, self.hd, 2, device=device).float() / self.hd))
            f = torch.outer(torch.arange(T, device=device).float(), inv)
            emb = torch.cat([f, f], -1)
            self._rope_cache[key] = (emb.cos(), emb.sin())
        return self._rope_cache[key]

    def hidden(self, ids):
        x = self.emb(ids)
        if self.emb_proj is not None:
            # in float32 even under autocast: the residual stream must stay float32, as it does when it starts
            # from a plain embedding lookup (a bf16 e @ P would make the whole stream bf16)
            with torch.autocast(device_type=ids.device.type, enabled=False):
                x = x @ self.emb_proj
        cos, sin = self.rope(ids.shape[1], ids.device)
        for blk in self.blocks:
            x = blk(x, cos.to(x.dtype), sin.to(x.dtype))
        return self.norm(x)

    def forward(self, ids):
        h = self.hidden(ids)
        if self.emb_proj is not None:
            h = h @ self.emb_proj.T
        return h @ self.emb.weight.T


def param_counts(model: ProxyLM) -> dict:
    """embedding = the word table plus, with emb_rank, its tied r x d widening matrix."""
    emb = model.emb.weight.numel()
    out = {}
    if model.emb_proj is not None:
        out["emb_proj"] = model.emb_proj.numel()
        emb += out["emb_proj"]
    total = sum(p.numel() for p in model.parameters())
    return {"total": total, "embedding": emb, "non_embedding": total - emb, **out}


# --------------------------------------------------------------------------------------------
# Data: packed uint16 4,096-token blocks of general text + biographies
# --------------------------------------------------------------------------------------------
def _blocks(source: str, split: str) -> np.ndarray:
    man = json.loads((VAL_ROOT / source / split / "manifest.json").read_text())
    arrs = [np.memmap(VAL_ROOT / source / split / s["path"], dtype="<u2", mode="r").reshape(-1, BLOCK)
            for s in man["shards"]]
    return arrs


class Stream:
    """Deterministic batches: seq-length windows from shuffled training blocks; some rows are fact windows."""

    def __init__(self, seq: int, batch: int, steps: int, fact_docs: list[list[int]], seed: int,
                 sources=("general",), max_fact_share: float | None = None):
        self.seq, self.batch = seq, batch
        self.rng = np.random.default_rng(seed)
        self.shards = [a for s in sources for a in _blocks(s, "train")]
        self.index = [(i, j) for i, a in enumerate(self.shards) for j in range(a.shape[0])]
        self.rng.shuffle(self.index)
        per_block = BLOCK // seq
        fact_tokens = []
        for d in fact_docs:
            fact_tokens += d + [EOS]
        n_fact_rows = math.ceil(len(fact_tokens) / seq)
        fact_tokens += [EOS] * (n_fact_rows * seq - len(fact_tokens))
        self.fact_rows = np.array(fact_tokens, dtype=np.int64).reshape(n_fact_rows, seq)
        total_rows = steps * batch
        cap = total_rows // 4 if max_fact_share is None else int(total_rows * max_fact_share)
        if n_fact_rows > cap:
            raise ValueError("too many fact tokens for this budget")
        self.is_fact = np.zeros(total_rows, dtype=bool)
        self.is_fact[self.rng.choice(total_rows, n_fact_rows, replace=False)] = True
        self.per_block = per_block
        self.fact_share = n_fact_rows / total_rows
        self._web = 0
        self._fact = 0
        self._row = 0

    def _web_row(self):
        b, w = divmod(self._web, self.per_block)
        i, j = self.index[b % len(self.index)]
        self._web += 1
        return np.asarray(self.shards[i][j, w * self.seq:(w + 1) * self.seq], dtype=np.int64)

    def next(self) -> np.ndarray:
        rows = []
        for _ in range(self.batch):
            if self.is_fact[self._row]:
                rows.append(self.fact_rows[self._fact]); self._fact += 1
            else:
                rows.append(self._web_row())
            self._row += 1
        return np.stack(rows)


# --------------------------------------------------------------------------------------------
# Probes
# --------------------------------------------------------------------------------------------
@torch.no_grad()
def val_loss(model, device, source: str, n_windows: int = 64, seq: int = 512) -> float:
    arrs = _blocks(source, "validation")
    allb = np.concatenate([np.asarray(a[: max(1, n_windows // len(arrs) + 1)]) for a in arrs], 0)
    rows = []
    for b in range(allb.shape[0]):
        for w in range(BLOCK // seq):
            rows.append(allb[b, w * seq:(w + 1) * seq])
    rng = np.random.default_rng(0)
    pick = rng.choice(len(rows), min(n_windows, len(rows)), replace=False)
    ids = torch.tensor(np.stack([rows[i] for i in sorted(pick)]).astype(np.int64), device=device)
    tot, n = 0.0, 0
    for i in range(0, ids.shape[0], 16):
        x = ids[i:i + 16]
        logits = model(x[:, :-1]).float()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction="sum")
        tot += ce.item(); n += x[:, 1:].numel()
    return tot / n


@torch.no_grad()
def fact_recall(model, device, tok, rows: list[dict]) -> list[bool]:
    """Teacher-forced greedy check: every answer token is the argmax after the prompt."""
    out = []
    for i in range(0, len(rows), 64):
        chunk = rows[i:i + 64]
        seqs, spans = [], []
        for r in chunk:
            p = [EOS] + tok.encode(r["prompt"], add_special_tokens=False).ids
            full = [EOS] + tok.encode(r["prompt"] + r["answer"], add_special_tokens=False).ids
            assert full[:len(p)] == p, r
            seqs.append(full); spans.append((len(p), len(full)))
        T = max(len(s) for s in seqs)
        ids = torch.tensor([s + [EOS] * (T - len(s)) for s in seqs], device=device)
        pred = model(ids).argmax(-1)
        for b, (a, z) in enumerate(spans):
            out.append(bool((pred[b, a - 1:z - 1] == ids[b, a:z]).all()))
    return out


# --------------------------------------------------------------------------------------------
# Train + probe one (recipe, seed)
# --------------------------------------------------------------------------------------------
def train_and_probe(recipe: dict, seed: int, out_dir: Path, device: str | None = None, log=print) -> dict:
    from tokenizers import Tokenizer
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(TOKENIZER))
    vocab = tok.get_vocab_size()
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)

    m = recipe["model"]
    model = ProxyLM(vocab, m["d"], m["layers"], m["heads"], m["kv"], m["ffn"], emb_rank=m.get("emb_rank")).to(device)
    counts = param_counts(model)
    t = recipe["train"]
    seq, batch, tokens = t["seq"], t["batch"], t["tokens"]
    steps = tokens // (seq * batch)

    people = factlib.make_people(recipe["facts"]["per_tier"], seed=recipe["facts"].get("seed", 0),
                                 tiers=tuple(recipe["facts"].get("tiers", factlib.TIERS)))
    docs = [tok.encode(d, add_special_tokens=False).ids for d in factlib.exposures(people, seed=seed)]
    stream = Stream(seq, batch, steps, docs, seed=seed, sources=tuple(t.get("sources", ["general"])),
                    max_fact_share=t.get("max_fact_share"))

    # muP-style: hidden-matrix LR scales with base_width / d; embeddings and norms use the base LR
    base_lr, base_w = t["lr"], t.get("lr_base_width", 256)
    hidden = [p for n, p in model.named_parameters() if p.ndim == 2 and not n.startswith("emb")]
    other = [p for n, p in model.named_parameters() if p.ndim < 2 or n.startswith("emb")]
    scale = base_w / m["d"] if t.get("mup_lr", True) else 1.0
    opt = torch.optim.AdamW([{"params": hidden, "lr": base_lr * scale, "base": base_lr * scale},
                             {"params": other, "lr": base_lr, "base": base_lr}],
                            betas=(0.9, 0.95), weight_decay=t.get("wd", 0.1), eps=1e-8)
    warm, decay_from = int(steps * t.get("warmup", 0.02)), int(steps * t.get("decay_start", 0.8))
    amp = torch.autocast(device_type=device, dtype=torch.bfloat16) if t.get("bf16", True) else _Null()
    log(f"recipe {recipe['id']} seed {seed}: {counts} steps {steps} fact share {stream.fact_share:.3%}")
    t0, hist = time.time(), []
    for step in range(steps):
        lr_mult = (step + 1) / max(1, warm) if step < warm else (
            1.0 if step < decay_from else max(0.1, 1 - 0.9 * (step - decay_from) / max(1, steps - decay_from)))
        for g in opt.param_groups:
            g["lr"] = g["base"] * lr_mult
        x = torch.from_numpy(stream.next()).to(device)
        with amp:
            logits = model(x[:, :-1])
        loss = F.cross_entropy(logits.float().reshape(-1, vocab), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            hist.append((step, round(loss.item(), 4)))
            if step % 500 == 0:
                log(f"  step {step}/{steps} loss {loss.item():.3f} {time.time() - t0:.0f}s")
    train_s = time.time() - t0

    model.eval()
    res = {"recipe": recipe["id"], "seed": seed, "params": counts, "steps": steps, "tokens": steps * seq * batch,
           "train_seconds": round(train_s, 1), "fact_share": stream.fact_share, "loss_history": hist,
           "val_loss": {s: val_loss(model, device, s) for s in ("general", "stack")}}
    for kind in ("seen", "unseen"):
        rows = factlib.probes(people, kind)
        ok = fact_recall(model, device, tok, rows)
        by = {}
        for r, o in zip(rows, ok):
            k = str(r["tier"])
            by.setdefault(k, [0, 0]); by[k][0] += o; by[k][1] += 1
        res[f"recall_{kind}"] = {k: v[0] / v[1] for k, v in sorted(by.items(), key=lambda kv: int(kv[0]))}
        if "tiers" in recipe["facts"]:   # E05+: counts per tier (absent before, so older results are unchanged)
            res[f"facts_{kind}_by_tier"] = {k: v[0] for k, v in sorted(by.items(), key=lambda kv: int(kv[0]))}
        res[f"facts_{kind}"] = int(sum(ok))
        res.setdefault("fact_total", len(rows))
    (out_dir / "result.json").write_text(json.dumps(res, indent=1))
    log(f"  done {recipe['id']} seed {seed}: val {res['val_loss']} seen {res['recall_seen']} ({train_s:.0f}s)")
    return res


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
