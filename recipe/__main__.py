"""python -m recipe freeze|run|report <ID>   (configs/<ID>.json; results in runs/recipe/<ID>/)."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m recipe", description=__doc__)
    ap.add_argument("command", choices=("freeze", "run", "report"))
    ap.add_argument("experiment")
    a = ap.parse_args(argv)
    from . import loop
    if a.command == "freeze":
        print(json.dumps(loop.freeze(a.experiment), indent=1))
    elif a.command == "run":
        log = lambda *x: print(*x, flush=True)
        if "tune" in loop.load_experiment(a.experiment):
            loop.run_tuned(a.experiment, log=log)
        else:
            loop.run(a.experiment, log=log)
        print(json.dumps(loop.report(a.experiment)["pairs"], indent=1))
    else:
        r = loop.report(a.experiment)
        for arm, s in r["summary"].items():
            print(arm, {k: v for k, v in s.items() if k != "params"})
        print(json.dumps(r["pairs"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
