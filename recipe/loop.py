"""The recipe loop: propose -> paired probe -> adopt -> ledger, applied to pretraining recipes.

Compared with an agent-harness loop:
  * the thing being edited is a training recipe (model shape, optimizer, data), not a prompt;
  * one probe is a whole training run, so the unit of replication is the seed, not the task;
  * arms are paired by seed (same init seed, data order and fact order), and an arm is adopted only
    when every seed pair agrees in sign and the mean gain clears a preregistered margin;
  * results are proxies: an adopted recipe is a proposal for a real training run, never a change to one.

Experiments live in configs/<ID>.json; results in runs/recipe/<ID>/<arm>/seed<n>/result.json.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import statistics
import time
from pathlib import Path

from .events import emit, runs_root

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def load_experiment(exp_id: str) -> dict:
    return json.loads((CONFIG_DIR / f"{exp_id}.json").read_text())


def arm_recipe(exp: dict, arm: str) -> dict:
    r = copy.deepcopy(exp["base"])
    for section, patch in exp["arms"][arm].items():
        r.setdefault(section, {}).update(patch)
    r["id"] = f"{exp['id']}/{arm}"
    return r


def out_dir(exp_id: str) -> Path:
    return runs_root() / "recipe" / exp_id


def freeze(exp_id: str) -> dict:
    p = CONFIG_DIR / f"{exp_id}.json"
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    d = out_dir(exp_id); d.mkdir(parents=True, exist_ok=True)
    rec = {"experiment": exp_id, "config": str(p), "sha256": sha, "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    f = d / "FROZEN.json"
    if f.exists():
        old = json.loads(f.read_text())
        if old["sha256"] != sha:
            raise SystemExit(f"{exp_id} config changed after freeze ({old['sha256'][:12]} -> {sha[:12]})")
        return old
    f.write_text(json.dumps(rec, indent=1))
    return rec


def run(exp_id: str, log=print) -> list[dict]:
    """Train every (seed, arm) pair not done yet; seeds outer so partial results stay paired."""
    exp = load_experiment(exp_id)
    if "tune" in exp:
        raise SystemExit(f"{exp_id} has a tune stage: use run_tuned")
    from .proxy import train_and_probe          # torch lives only here
    fr = freeze(exp_id)
    if fr["sha256"] != hashlib.sha256((CONFIG_DIR / f"{exp_id}.json").read_bytes()).hexdigest():
        raise SystemExit("frozen config mismatch")
    done = []
    ledger = out_dir(exp_id) / "ledger.jsonl"
    for seed in exp["seeds"]:
        for arm in exp["arms"]:
            d = out_dir(exp_id) / arm / f"seed{seed}"
            if (d / "result.json").exists():
                done.append(json.loads((d / "result.json").read_text()))
                continue
            emit("recipe_start", experiment=exp_id, arm=arm, seed=seed)
            res = train_and_probe(arm_recipe(exp, arm), seed, d, log=log)
            res["arm"] = arm
            (d / "result.json").write_text(json.dumps(res, indent=1))
            with ledger.open("a") as f:
                f.write(json.dumps({"t": time.time(), "arm": arm, "seed": seed, "facts_seen": res["facts_seen"],
                                    "val_general": res["val_loss"]["general"],
                                    "train_seconds": res["train_seconds"]}) + "\n")
            emit("recipe_end", experiment=exp_id, arm=arm, seed=seed, facts_seen=res["facts_seen"],
                 val_general=round(res["val_loss"]["general"], 4))
            done.append(res)
    return done


def _canon(recipe: dict) -> str:
    r = {k: v for k, v in recipe.items() if k != "id"}
    return json.dumps(r, sort_keys=True)


def with_lr_mult(recipe: dict, mult: float) -> dict:
    r = copy.deepcopy(recipe)
    r["train"]["lr"] = r["train"]["lr"] * mult
    r["id"] = f"{recipe['id']}@lr{mult:g}"
    return r


def _reuse_candidates(other_id: str, seed: int):
    """(result file, the recipe that produced it) for every finished run of an earlier experiment at this seed.
    A tuned experiment's finals ran at the multiplier in its selection.json, and its sweep runs at their own."""
    other = load_experiment(other_id)
    root = out_dir(other_id)
    picks = {}
    if "tune" in other and (root / "selection.json").exists():
        picks = json.loads((root / "selection.json").read_text())
    for arm in other["arms"]:
        base = arm_recipe(other, arm)
        if "tune" not in other:
            yield root / arm / f"seed{seed}" / "result.json", base
            continue
        if arm in picks:
            yield root / arm / f"seed{seed}" / "result.json", with_lr_mult(base, picks[arm]["mult"])
            for m in picks[arm]["tried"]:
                yield (root / "sweep" / f"{arm}_lr{float(m):g}" / f"seed{seed}" / "result.json",
                       with_lr_mult(base, float(m)))


def _find_reuse(exp: dict, recipe: dict, seed: int):
    """A finished run in an earlier experiment with a byte-identical recipe (ignoring its id) and the same seed."""
    want = _canon(recipe)
    for other_id in exp.get("tune", {}).get("reuse_from", []):
        for f, r in _reuse_candidates(other_id, seed):
            if f.exists() and _canon(r) == want:
                return f
    return None


def _score(res: dict, metric: str) -> float:
    v = _get(res, metric)
    return math.inf if v is None or (isinstance(v, float) and math.isnan(v)) else v


def _run_one(exp: dict, recipe: dict, seed: int, d: Path, train, log, tag: str) -> dict:
    f = d / "result.json"
    if f.exists():
        return json.loads(f.read_text())
    src = _find_reuse(exp, recipe, seed)
    if src is not None:
        d.mkdir(parents=True, exist_ok=True)
        res = json.loads(src.read_text())
        res["reused_from"] = str(src)
        f.write_text(json.dumps(res, indent=1))
        log(f"{tag}: reused identical run {src}")
        return res
    emit("recipe_start", experiment=exp["id"], run=tag, seed=seed)
    d.mkdir(parents=True, exist_ok=True)
    res = train(recipe, seed, d, log=log)
    f.write_text(json.dumps(res, indent=1))
    with (out_dir(exp["id"]) / "ledger.jsonl").open("a") as g:
        g.write(json.dumps({"t": time.time(), "run": tag, "seed": seed, "lr": recipe["train"]["lr"],
                            "facts_seen": res["facts_seen"], "val_general": res["val_loss"]["general"],
                            "train_seconds": res["train_seconds"]}) + "\n")
    emit("recipe_end", experiment=exp["id"], run=tag, seed=seed, facts_seen=res["facts_seen"])
    return res


def run_tuned(exp_id: str, log=print, train=None) -> dict:
    """Stage 1: each arm at every LR multiplier on the tune seed; pick the lowest select_by (never the primary).
    If the pick is at an edge of the grid, try one more point beyond it (x0.5 or x2) and pick again.
    Stage 2: each arm at its picked multiplier on the final seeds, which exclude the tune seed."""
    if train is None:
        from .proxy import train_and_probe as train
    exp = load_experiment(exp_id)
    fr = freeze(exp_id)
    if fr["sha256"] != hashlib.sha256((CONFIG_DIR / f"{exp_id}.json").read_bytes()).hexdigest():
        raise SystemExit("frozen config mismatch")
    t = exp["tune"]
    assert t["tune_seed"] not in exp["seeds"], "final seeds must exclude the tune seed"
    picks = {}
    for arm in exp["arms"]:
        base = arm_recipe(exp, arm)
        tried = {}
        def trial(m):
            d = out_dir(exp_id) / "sweep" / f"{arm}_lr{m:g}" / f"seed{t['tune_seed']}"
            tried[m] = _score(_run_one(exp, with_lr_mult(base, m), t["tune_seed"], d, train, log,
                                       f"sweep {arm} x{m:g}"), t["select_by"])
        for m in t["lr_mults"]:
            trial(m)
        best = min(tried, key=lambda m: (tried[m], m))
        extended = None
        if t.get("edge_extend"):
            grid = sorted(t["lr_mults"])
            if best == grid[0]:
                extended = best / 2
            elif best == grid[-1]:
                extended = best * 2
            if extended is not None:
                trial(extended)
                best = min(tried, key=lambda m: (tried[m], m))
        picks[arm] = {"mult": best, "lr": base["train"]["lr"] * best, "tried": {f"{m:g}": v for m, v in sorted(tried.items())},
                      "extended": extended, "edge": best in (min(tried), max(tried))}
        log(f"pick {arm}: x{best:g} ({picks[arm]['tried']})")
    (out_dir(exp_id) / "selection.json").write_text(json.dumps(picks, indent=1))
    for seed in exp["seeds"]:
        for arm in exp["arms"]:
            r = with_lr_mult(arm_recipe(exp, arm), picks[arm]["mult"])
            res = _run_one(exp, r, seed, out_dir(exp_id) / arm / f"seed{seed}", train, log, f"final {arm} seed {seed}")
            res["arm"] = arm
    return picks


def _get(res: dict, metric: str):
    v = res
    for k in metric.split("."):
        v = v[k]
    return v


def report(exp_id: str) -> dict:
    exp = load_experiment(exp_id)
    rule = exp["decision"]
    rows: dict[str, dict[int, dict]] = {}
    for arm in exp["arms"]:
        for seed in exp["seeds"]:
            f = out_dir(exp_id) / arm / f"seed{seed}" / "result.json"
            if f.exists():
                rows.setdefault(arm, {})[seed] = json.loads(f.read_text())
    summary = {}
    for arm, by_seed in rows.items():
        def col(m):
            return [_get(r, m) for r in by_seed.values()]
        s = {"seeds": sorted(by_seed)}
        for m in (rule["primary"], rule["guard"], "facts_unseen", "val_loss.stack", "train_seconds"):
            vals = col(m)
            s[m] = {"mean": statistics.mean(vals), "sd": statistics.stdev(vals) if len(vals) > 1 else None}
        tiers = {}
        for r in by_seed.values():
            for k, v in r["recall_seen"].items():
                tiers.setdefault(k, []).append(v)
        s["recall_seen_by_tier"] = {k: statistics.mean(v) for k, v in tiers.items()}
        s["params"] = next(iter(by_seed.values()))["params"]
        summary[arm] = s
    total = None
    pairs = {}
    arms = list(rows)
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            common = sorted(set(rows[a]) & set(rows[b]))
            if not common:
                continue
            total = total or rows[a][common[0]]["fact_total"]
            d = [_get(rows[a][s], rule["primary"]) - _get(rows[b][s], rule["primary"]) for s in common]
            g = [_get(rows[a][s], rule["guard"]) - _get(rows[b][s], rule["guard"]) for s in common]
            margin = rule["margin_frac"] * total
            if len(common) < len(exp["seeds"]):
                verdict = "INCOMPLETE"
            elif all(x >= margin for x in d) or (statistics.mean(d) >= margin and all(x > 0 for x in d)):
                verdict = f"{a.upper()}_STORES_MORE"
            elif all(x <= -margin for x in d) or (statistics.mean(d) <= -margin and all(x < 0 for x in d)):
                verdict = f"{b.upper()}_STORES_MORE"
            else:
                verdict = "TIE"
            if verdict.endswith("STORES_MORE"):
                win_is_a = verdict.startswith(a.upper())
                guard_cost = statistics.mean(g) if win_is_a else -statistics.mean(g)
                if guard_cost > rule["guard_margin"]:
                    verdict += "_BUT_WORSE_LOSS"
            pairs[f"{a} vs {b}"] = {"seeds": common, "diff_primary": d, "mean_diff": statistics.mean(d),
                                    "margin": margin, "diff_guard": g, "verdict": verdict}
    out = {"experiment": exp_id, "summary": summary, "pairs": pairs, "rule": rule}
    sel = out_dir(exp_id) / "selection.json"
    if sel.exists():
        out["lr_selection"] = json.loads(sel.read_text())
    (out_dir(exp_id) / "report.json").write_text(json.dumps(out, indent=1))
    return out
