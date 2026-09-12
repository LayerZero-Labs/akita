# Exact JL budget planning tool

`jl_budget_factory.py` audits a finite lower/upper CertifiedJL endpoint registry
and searches level allocations without floating-point admission decisions. It is
a planning tool, not a certificate checker or production geometry validator.

Run the checked census and sensitivity examples:

```bash
python3 scripts/jl_budget_factory.py audit \
  scripts/jl_budget_planning_example.json
```

Run bounded exact level searches under two configurable budgets:

```bash
python3 scripts/jl_budget_factory.py search \
  scripts/jl_budget_planning_example.json --budget-bits 136
python3 scripts/jl_budget_factory.py search \
  scripts/jl_budget_planning_example.json --budget-bits 132
```

The JSON output preserves all probabilities and distortion products as exact
`numerator`/`denominator` pairs. A search rejects an unexpectedly large product
instead of silently sampling it; narrow `searchPairs` or explicitly raise
`--max-combinations`. `--max-results` only truncates rendered Pareto members,
and the report retains the full frontier size and a truncation flag.

The v1 input separates `frontier.lower` and `frontier.upper`. Every endpoint
names its own exact rational or dyadic failure cap and records its rational
threshold, row law/count, status, theorem, source revision, and tail-specific
modulus hypotheses. A scenario can select endpoints by fold level or by logical
projection use. The exact budget check is

```text
sum_level candidates_level
  * sum_use blocks_use * (delta_lower_use + delta_upper_use)
  <= failureBudget.
```

Matrix-envelope sharing is audited separately from this ledger. The example
has 46 logical use rows sharing 20 `(fold, depth)` envelopes, but all 5,548
block applications and 22,192 candidate-weighted lower/upper opportunities
remain charged. Declared dependency paths must increase strictly in depth;
joins must occur after every path they extend; and an envelope cannot cross a
fold/depth boundary.

Level search first removes locally dominated `(failure, U/L)` endpoint pairs,
then retains the full vector of exact path `U/L` products as its geometry
objective. It does not replace that vector with a worst-path or invented width
heuristic, and it makes no claim that uniform bit allocations are optimal.
Per-use allocations are fully auditable; per-use search is available only when
the explicit finite combination limit permits exhaustive enumeration.

The committed example is labeled `untrusted-exploration`. Its intermediate
geometry is synthetic and must be replaced by theorem- and revision-pinned
CertifiedJL records. Reports from this tool always set `productionAdmissible`
to false because proof replay, modulus feasibility, and production registry
admission are outside its scope.

## Example with kernel-checked endpoint records

`jl_budget_certified_frontier_example.json` uses nine rational lower endpoints
at 144 through 152 bits and thirteen rational upper endpoints at 140 through
152 bits. Every record names the exact CertifiedJL declaration and source
commit. Its projection forest remains the exploratory nv30 census; the tool
does not certify its modulus conditions or structured no-wrap caps.

```bash
python3 scripts/jl_budget_factory.py audit \
  scripts/jl_budget_certified_frontier_example.json
```

The three supplied assignments use the same `2^-136` JL budget:

| Assignment | Exact budget fraction used | Largest elementary path distortion |
| --- | ---: | ---: |
| Uniform 151 bits | `1387/2048` | about 109,268 |
| Per-fold `[151,151,148,146,146,145]` | `4093/4096` | about 109,268 |
| Per-use improvement with independent tails | `1` | about 85,452 |

The last assignment decreases every path's product of `U/L` ratios relative
to uniform 151, with about 21.8% improvement on the worst path. Its selection
method is recorded in the scenario. This is a feasible local improvement over
the listed endpoints, not a globally optimal schedule. The path products omit
the structured block-allocation and join factors needed to derive actual
no-wrap caps; production admission must include those factors separately.
