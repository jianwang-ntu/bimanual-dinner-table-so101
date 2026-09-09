#!/usr/bin/env python3
"""Compare eval_seeds runs seed-for-seed and sub-goal-for-sub-goal.

A ten-seed total is a weak instrument: this project has already measured that
cells differing only in controller knobs run 10-19 sub-goals with sd 2.0, so a
two- or three-point move in the total is inside the sweep's own spread.  What is
NOT inside that spread is a named seed gaining a named sub-goal that a written
prediction said it would gain.  So this prints both, and prints the per-seed
delta first.

Run:  PYTHONPATH=. python3 scripts/compare_eval_cells.py \
          base=evidence/ab_mug_control_shipped.json \
          f155=evidence/ab_mug_frac155.json
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ORDER = ["drawer_open", "fork_placed", "spoon_placed", "plate_placed",
         "mug_placed"]


def load(p: str) -> dict:
    d = json.loads((ROOT / p).read_text())
    return {e["seed"]: e["task"]["subgoals"] for e in d["episodes"]}


def main() -> int:
    cells = {}
    for arg in sys.argv[1:]:
        name, _, path = arg.partition("=")
        if not path:
            print(f"expected name=path, got {arg!r}", file=sys.stderr)
            return 2
        cells[name] = load(path)
    if len(cells) < 2:
        print("need at least two cells", file=sys.stderr)
        return 2

    names = list(cells)
    base = names[0]
    seeds = sorted(cells[base])

    print(f"{'seed':<5}" + "".join(f"{n:<22}" for n in names))
    for s in seeds:
        row = f"{s:<5}"
        for n in names:
            sg = cells[n][s]
            row += "".join("1" if sg[g] else "." for g in ORDER) + " "
            row += f"({sum(sg.values())})".ljust(22 - len(ORDER) - 1)
        print(row)
    print(f"      order: {'/'.join(g.split('_')[0] for g in ORDER)}")
    print()

    print(f"{'subgoal':<16}" + "".join(f"{n:>10}" for n in names)
          + "   delta vs " + base)
    for g in ORDER:
        counts = {n: sum(1 for s in seeds if cells[n][s][g]) for n in names}
        d = "  ".join(f"{n}:{counts[n] - counts[base]:+d}"
                      for n in names[1:])
        print(f"{g:<16}" + "".join(f"{counts[n]:>10}" for n in names)
              + f"   {d}")
    tot = {n: sum(sum(cells[n][s].values()) for s in seeds) for n in names}
    print(f"{'TOTAL':<16}" + "".join(f"{tot[n]:>10}" for n in names)
          + "   " + "  ".join(f"{n}:{tot[n] - tot[base]:+d}" for n in names[1:]))
    succ = {n: sum(1 for s in seeds if all(cells[n][s].values())) for n in names}
    print(f"{'task_success':<16}" + "".join(f"{succ[n]:>10}" for n in names))
    print()

    for n in names[1:]:
        gained = [(s, g) for s in seeds for g in ORDER
                  if cells[n][s][g] and not cells[base][s][g]]
        lost = [(s, g) for s in seeds for g in ORDER
                if cells[base][s][g] and not cells[n][s][g]]
        print(f"{n} vs {base}:")
        print(f"  gained: {gained or 'none'}")
        print(f"  lost  : {lost or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
