"""ProxyLM with and without emb_rank (needs torch; skipped in the stdlib venv)."""
import importlib.util
import math
import os
import subprocess

import pytest

torch = pytest.importorskip("torch")
from recipe.proxy import ProxyLM, param_counts  # noqa: E402

SHAPE = dict(vocab=97, d=32, layers=2, heads=4, kv=2, ffn=48)


def _model(**kw):
    torch.manual_seed(0)
    return ProxyLM(**{**SHAPE, **kw})


def test_default_path_is_the_committed_model():
    """No emb_rank: same parameters, same init and same logits as the model at git HEAD~ (the E02/E03 code)."""
    ref = os.environ.get("RECIPE_PROXY_REF")
    if not ref:
        pytest.skip("set RECIPE_PROXY_REF to a copy of the pre-E04 proxy.py")
    spec = importlib.util.spec_from_file_location("recipe.proxy_ref", ref)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    torch.manual_seed(0); old = mod.ProxyLM(**SHAPE)
    new = _model()
    assert [n for n, _ in old.named_parameters()] == [n for n, _ in new.named_parameters()]
    for (n, a), (_, b) in zip(old.named_parameters(), new.named_parameters()):
        assert torch.equal(a, b), n
    ids = torch.randint(0, 97, (2, 9), generator=torch.Generator().manual_seed(1))
    assert torch.equal(old(ids), new(ids))
    assert param_counts(new) == mod.param_counts(old)


def test_emb_rank_equal_to_d_is_the_default():
    a, b = _model(), _model(emb_rank=32)
    assert b.emb_proj is None
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))


def test_emb_rank_shapes_counts_and_scale():
    m = _model(emb_rank=8)
    assert m.emb.weight.shape == (97, 8) and m.emb_proj.shape == (8, 32)
    c = param_counts(m)
    assert c["emb_proj"] == 8 * 32 and c["embedding"] == 97 * 8 + 8 * 32
    assert c["total"] == c["embedding"] + c["non_embedding"]
    full = param_counts(_model())
    assert c["non_embedding"] == full["non_embedding"]                  # the blocks are untouched
    # orthonormal rows scaled by sqrt(d/r): P P^T = (d/r) I
    assert torch.allclose(m.emb_proj @ m.emb_proj.T, torch.eye(8) * (32 / 8), atol=1e-4)
    ids = torch.randint(0, 97, (2, 9))
    assert m(ids).shape == (2, 9, 97)
    with pytest.raises(ValueError):
        _model(emb_rank=33)


def test_emb_rank_trains_on_cpu():
    m = _model(emb_rank=8)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    ids = torch.randint(0, 97, (4, 17), generator=torch.Generator().manual_seed(2))
    first = None
    for _ in range(30):
        logits = m(ids[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 97), ids[:, 1:].reshape(-1))
        first = first if first is not None else loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first - 0.5 and m.emb_proj.grad is not None


def test_residual_stream_stays_float32_under_autocast():
    """The E04 path must keep the residual stream float32 under bf16 autocast, like the plain lookup does."""
    seen = {}
    for name, m in (("plain", _model()), ("rank", _model(emb_rank=8))):
        blk = m.blocks[0]
        h = blk.register_forward_pre_hook(lambda mod, args, name=name: seen.__setitem__(name, args[0].dtype))
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            m(torch.randint(0, 97, (2, 9)))
        h.remove()
    assert seen == {"plain": torch.float32, "rank": torch.float32}


def test_fact_share_cap_default_and_raised():
    from recipe import proxy
    docs = [[5] * 511] * 30                       # 30 fact rows of 512 at seq 512
    with pytest.raises(ValueError):               # default cap: a quarter of 2 x 50 = 100 rows -> 25 < 30
        proxy.Stream.__new__(proxy.Stream).__init__(512, 2, 50, docs, seed=0, sources=())
    s = proxy.Stream.__new__(proxy.Stream)
    s.__init__(512, 2, 50, docs, seed=0, sources=(), max_fact_share=0.35)
    assert s.fact_share == pytest.approx(0.30)
