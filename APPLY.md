# Landing this branch

The bundle carries the full branch — 8 commits, `research/jpm-vol-surface-replication` — with
history intact. Authorship on all commits is `Vish Shah <vish.shah@sabacapital.com>`; amend if
you'd rather they read otherwise.

## Into an existing repo

```bash
cd /path/to/your/repo
git fetch /path/to/volsurface-pr.bundle research/jpm-vol-surface-replication
git checkout -b research/jpm-vol-surface-replication FETCH_HEAD
git push -u origin research/jpm-vol-surface-replication
```

Then open the PR and paste `PR_DESCRIPTION.md` as the body.

If the repo has an existing layout and `src/volsurface/` would collide, rewrite the prefix before
pushing:

```bash
git filter-repo --path-rename src/volsurface/:src/research/volsurface/
```

## As a standalone repo

```bash
git clone -b research/jpm-vol-surface-replication /path/to/volsurface-pr.bundle volsurface
cd volsurface
git remote set-url origin <your-remote>
git push -u origin research/jpm-vol-surface-replication
```

## Verify before pushing

```bash
pip install -e .
pytest                       # 11 passing
python scripts/run_smoke.py  # both strategies end to end, synthetic surface
```

## Commit sequence

Each commit is one reviewable unit; the message body carries the reasoning, so
`git log` is the design doc.

| | |
|---|---|
| `9082f28` | scaffolding + Black-Scholes core |
| `ed234ca` | VolSurface protocol, OptionMetrics + synthetic adapters |
| `cd7381f` | delta-hedged trade engine with transaction costs |
| `46ca0ad` | regime signals and performance metrics |
| `bdfe757` | strategies #1 and #2 |
| `4d92838` | sensitivity, walk-forward, cost ladder |
| `92401db` | tests and smoke run |
| `4c5e11d` | README and PR description |

Tests land at `92401db`, so the five commits before it won't have a green suite in isolation.
Squash-merge if your CI gates per-commit.
