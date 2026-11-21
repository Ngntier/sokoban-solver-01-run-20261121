# Sokoban Solver — make.fun bounty entry

A push-optimal Sokoban solver for the "Build the best Sokoban solver, 21 November 2026" bounty ($100 in USDC).

## How it works

The generator builds every level as a hallway with independent rooms (3–4 boxes
each) off it, and a level's push-optimum P* is the sum of its rooms' optima.
So each room is solved push-optimally on its own, the rooms in turn, the
player walking between them. Pushes, not moves, are what the score counts: a
level solved in P* pushes earns 1,000 points; one push over earns about half.

- **Room decomposition**: the dead squares of the player's row (the hallway:
  a box there reaches no goal) are cut out of the floor; each remaining part
  holding boxes is a room.
- **A\* over pushes**: a state is the boxes' squares plus the player's
  reachable region (named by its lowest square). The estimate is the cheapest
  assignment of boxes to goals by push-distance (a minimum matching), which is
  admissible, so the first solution found is push-optimal.
- **Deadlock pruning**: dead squares (a box that can never reach a goal),
  frozen 2×2 blocks, and frozen boxes (fixpoint).
- **Deterministic**: fixed expansion budgets, no clock, no randomness, ties
  broken by insertion order — two runs write the same bytes.

## Results (local, kit scorer)

- Practice set: **1,000,000** (all 5 instances at P*: 13, 16, 29, 47, 60
  pushes), ~0.4 s per run.
- 13 further sets drawn with the kit generator: **1,000,000** on every one,
  deterministic, all under 0.7 s per set (budget: 60 s).
- House baseline (plain-astar 1.0.0) scores 200,000 on the practice set
  (solves only the first instance); this entry beats it on every set.

## Run

```sh
MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads `/input/<id>/level.txt` (or `$MAKEFUN_INPUT_DIR`), writes
`/output/<id>/solution.txt` for every instance.

## Files

- `main.ts` — the solver (single file, Node 22, no dependencies).
- `LICENSE` — MIT.
