"""Recipe loop selection logic with a fake trainer (no torch)."""
import json

import pytest

from recipe import loop


def _fake_exp(tmp_path, monkeypatch, best):
    cfg = tmp_path / "cfg"; cfg.mkdir()
    base = {"model": {"d": 64}, "train": {"lr": 0.01}, "facts": {"per_tier": 1}}
    prev = {"id": "P", "base": base, "arms": {"a": {"model": {"d": 64}}, "b": {"model": {"d": 32}}}, "seeds": [0, 1],
            "decision": {}}
    exp = dict(prev, id="X", tune={"lr_mults": [0.5, 1, 2], "tune_seed": 0, "select_by": "val_loss.general",
                                   "edge_extend": True, "reuse_from": ["P"]}, seeds=[1, 2])
    (cfg / "P.json").write_text(json.dumps(prev)); (cfg / "X.json").write_text(json.dumps(exp))
    monkeypatch.setattr(loop, "CONFIG_DIR", cfg)
    monkeypatch.setenv("RECIPE_RUNS", str(tmp_path / "runs"))
    calls = []

    def train(recipe, seed, d, log=None):
        arm = "a" if recipe["model"]["d"] == 64 else "b"
        lr = recipe["train"]["lr"]
        calls.append((arm, round(lr / 0.01, 4), seed))
        return {"val_loss": {"general": abs(lr / 0.01 - best[arm]) + seed * 0.01}, "facts_seen": 1,
                "train_seconds": 1.0}
    return train, calls


def test_tuned_selection_edge_extension_and_reuse(tmp_path, monkeypatch):
    train, calls = _fake_exp(tmp_path, monkeypatch, best={"a": 1.0, "b": 0.25})
    # an identical earlier run (arm a, x1, seed 0) is reused, not retrained
    d = tmp_path / "runs" / "recipe" / "P" / "a" / "seed0"; d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({"val_loss": {"general": 0.0}, "facts_seen": 9, "train_seconds": 1}))
    picks = loop.run_tuned("X", log=lambda *a: None, train=train)
    assert picks["a"]["mult"] == 1 and picks["a"]["extended"] is None
    assert picks["b"]["mult"] == 0.25 and picks["b"]["extended"] == 0.25      # edge 0.5 -> tried 0.25
    assert ("a", 1.0, 0) not in calls                                         # reused
    assert ("b", 0.25, 0) in calls
    finals = [c for c in calls if c[2] in (1, 2)]
    assert sorted(finals) == [("a", 1.0, 1), ("a", 1.0, 2), ("b", 0.25, 1), ("b", 0.25, 2)]
    assert 0 not in [c[2] for c in finals]                                    # tune seed never in the finals


def test_plain_run_refuses_tuned_experiment(tmp_path, monkeypatch):
    _fake_exp(tmp_path, monkeypatch, best={"a": 1.0, "b": 1.0})
    with pytest.raises(SystemExit):
        loop.run("X")


def test_reuse_from_tuned_experiment_sweeps_and_finals(tmp_path, monkeypatch):
    """E04-style reuse: an arm identical to a tuned experiment's arm reuses both its sweep runs and its finals."""
    train, calls = _fake_exp(tmp_path, monkeypatch, best={"a": 1.0, "b": 0.25})
    loop.run_tuned("X", log=lambda *a: None, train=train)              # X is now a finished tuned experiment
    cfg = loop.CONFIG_DIR
    x = json.loads((cfg / "X.json").read_text())
    y = dict(x, id="Y", tune=dict(x["tune"], reuse_from=["X"]),
             arms={"a": {"model": {"d": 64, "emb_rank": 16}}, "b": {"model": {"d": 32}}})
    (cfg / "Y.json").write_text(json.dumps(y))
    calls.clear()
    picks = loop.run_tuned("Y", log=lambda *a: None, train=train)
    assert picks["b"]["mult"] == 0.25
    assert [c for c in calls if c[0] == "b"] == []                     # every b run (sweep + finals) reused
    assert {c[2] for c in calls if c[0] == "a"} == {0, 1, 2}            # the changed arm trains
    f = tmp_path / "runs" / "recipe" / "Y" / "b" / "seed1" / "result.json"
    assert json.loads(f.read_text())["reused_from"].endswith("X/b/seed1/result.json")
    s = tmp_path / "runs" / "recipe" / "Y" / "sweep" / "b_lr0.25" / "seed0" / "result.json"
    assert json.loads(s.read_text())["reused_from"].endswith("X/sweep/b_lr0.25/seed0/result.json")


def test_tier_pairs_and_exposures_to_recall(tmp_path, monkeypatch):
    """E05: the primary rule per tier (margin on that tier's facts), and the log-interpolated exposure crossing."""
    cfg = tmp_path / "cfg"; cfg.mkdir()
    exp = {"id": "T", "base": {"facts": {"per_tier": 10}}, "arms": {"w": {}, "d": {}}, "seeds": [1, 2, 3],
           "decision": {"primary": "facts_seen_by_tier.64", "guard": "val_loss.general", "margin_frac": 0.03,
                        "margin_total": 40, "guard_margin": 0.02, "tier_pairs": True}}
    (cfg / "T.json").write_text(json.dumps(exp))
    monkeypatch.setattr(loop, "CONFIG_DIR", cfg)
    monkeypatch.setenv("RECIPE_RUNS", str(tmp_path / "runs"))
    curves = {"w": {"1": 0.0, "4": 0.5, "64": 1.0}, "d": {"1": 0.0, "4": 0.25, "64": 0.75}}
    for arm, curve in curves.items():
        for s in (1, 2, 3):
            d = tmp_path / "runs" / "recipe" / "T" / arm / f"seed{s}"; d.mkdir(parents=True)
            counts = {k: round(v * 40) for k, v in curve.items()}
            (d / "result.json").write_text(json.dumps({
                "val_loss": {"general": 4.0, "stack": 6.0}, "facts_unseen": 0, "train_seconds": 1.0, "params": {},
                "recall_seen": curve, "facts_seen_by_tier": counts, "fact_total": 120}))
    r = loop.report("T")
    assert r["pairs"]["w vs d"]["margin"] == pytest.approx(1.2)                # 3% of margin_total 40
    assert r["pairs"]["w vs d"]["verdict"] == "W_STORES_MORE"                  # tier 64: 40 vs 30 on every seed
    tp = r["tier_pairs"]["w vs d"]
    assert tp["1"]["verdict"] == "TIE" and tp["4"]["verdict"] == "W_STORES_MORE"
    assert r["exposures_to_recall"]["w"]["0.5"] == 4                           # reaches 0.5 exactly at tier 4
    # d: 0.25 at 4, 0.75 at 64 -> 0.5 halfway in log2 between 4 and 64 = 2**4 = 16
    assert r["exposures_to_recall"]["d"]["0.5"] == pytest.approx(16.0)
    assert r["exposures_to_recall"]["d"]["0.9"] is None


def test_extra_tiers_keep_the_first_people():
    from recipe import facts
    base = facts.make_people(120, seed=0)
    assert base == facts.make_people(120, seed=0, tiers=facts.TIERS)
    more = facts.make_people(120, seed=0, tiers=(1, 4, 16, 64, 256, 1024))
    assert more[:480] == base and len(more) == 720 and {p.tier for p in more[480:]} == {256, 1024}
    assert len({p.name for p in more}) == 720
