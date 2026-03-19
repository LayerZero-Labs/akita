# NIST-Style Security Checklist For Hachi

This note turns "follow NIST best practice" into a concrete checklist for Hachi parameter work.

The main lesson from the ML-KEM / ML-DSA standardization path is not "use one blessed lattice estimator." The actual NIST methodology is broader:

- state the relevant security claim first,
- use broad security floors rather than one exact bit number,
- evaluate more than one concrete attack view,
- include operational assumptions and malformed-input handling in the security story,
- choose defaults with visible reserve margin,
- and publish a reproducible dossier rather than a single headline estimate.

For Hachi, the analogue is not KEM or signature security. The right translation is:

- commitment binding,
- opening / extraction soundness,
- recursive reduction soundness,
- Fiat-Shamir transcript soundness,
- and the concrete SIS / MSIS hardness that supports the binding side.

Main repo anchors for the current tree:

- Current 128-bit prime:
  `src/algebra/fields/fp128.rs:900-901`
- Current 128-bit commitment profiles and the halving-D profile:
  `src/protocol/commitment/config.rs:663-814`
- Transcript labels and domain separation:
  `src/protocol/transcript/labels.rs:7-130`
- Sparse challenge sampling context binding:
  `src/protocol/challenges/sparse.rs:36-46`
- In-tree Euclidean Module-SIS mirror:
  `src/protocol/labrador/config.rs:116-274`
- Current profile-style SIS summary test:
  `src/protocol/labrador/config.rs:990-1043`
- Structural proof validation for Labrador payloads:
  `src/protocol/proof.rs:914-942`
- Existing Hachi SIS note:
  `HACHI_SIS_ESTIMATION.md:1-705`

## What To Copy From NIST

The best NIST habit to copy is not a particular BKZ formula. It is the way the whole parameter dossier is organized.

1. Start from the attack game, not from the estimator.
   For NIST, the KEM target was `IND-CCA2` and the signature target was `EUF-CMA`, with concrete query bounds and explicit attention to classical oracle queries. For Hachi, the equivalent first step is to say exactly which claims are being made: binding only, soundness, knowledge soundness, or some larger end-to-end statement.

2. Treat "128-bit security" as a floor, not as one exact estimate.
   NIST categories are deliberately coarse. The point is to avoid pretending that one concrete lattice estimate is exact. For Hachi, the right translation is to publish a target floor for the total protocol, then make sure each major component comfortably clears it.

3. Check multiple concrete models.
   NIST accepted Kyber and Dilithium based on an overall body of evidence, even though some RAM-style estimates came out weaker than the headline categories when memory access cost was ignored. The lesson is: do not let one modeling choice completely drive the conclusion.

4. Include system conditions in the claim.
   NIST folds randomness requirements, failure behavior, side-channel posture, and invalid-input handling into the standard itself. For Hachi, transcript binding, domain separation, proof-shape validation, and challenge-sampling assumptions belong in the same parameter dossier as SIS hardness.

5. Pick defaults with reserve margin.
   NIST did not merely standardize the smallest acceptable set. It also chose practical defaults with visible headroom. For Hachi, the same idea means not treating "barely above 128" as a comfortable default when the overall protocol spends soundness elsewhere.

6. Document reevaluation triggers.
   NIST explicitly assumes reevaluation as cryptanalysis improves. Hachi should do the same whenever the attack model, challenge family, recursive reduction, or estimator defaults change.

## Hachi-Adapted Checklist

Below is the concrete checklist I would use for Hachi parameter selection.

### 1. State the exact security claim

Question:

- What exactly is this parameter set supposed to guarantee?

For Hachi, the minimum useful split is:

- challenge entropy,
- commitment binding hardness,
- protocol soundness terms outside SIS,
- and any extraction / knowledge claim if one is being made.

Current status: `partial`.

What is already good:

- `HACHI_SIS_ESTIMATION.md:7-13` already separates challenge entropy, witness geometry, and SIS binding hardness instead of conflating them.

What is still missing:

- One short note that says, in one place, what "Hachi-128" would actually mean for the full protocol and which terms contribute to that total budget.

### 2. Freeze the parameter menu before estimating

Question:

- Which concrete parameter families are actually under consideration?

Current status: `good`, but not yet finalized as a public menu.

What is already good:

- The repo already exposes a compact menu:
  - `Fp128FullCommitmentConfig`
  - `Fp128OneHotCommitmentConfig`
  - `Fp128LogBasisCommitmentConfig`
  - `Fp128HalvingDCommitmentConfig`
  via `src/protocol/commitment/config.rs:679-814`.

What is still missing:

- A written classification of which sets are:
  - recommended defaults,
  - plausible experimental variants,
  - research-only shells.

### 3. Derive geometry from live code, not from stale paper formulas

Question:

- Are the SIS / MSIS instances derived from the exact current layout logic?

Current status: `good`.

What is already good:

- `HACHI_SIS_ESTIMATION.md:76-78` explicitly says exact current-Hachi numbers should come from the live layout formulas in `src/protocol/commitment/config.rs` and `src/protocol/ring_switch.rs`.
- The in-tree Euclidean estimator mirror in `src/protocol/labrador/config.rs:116-274` flattens rank and width from the actual current geometry.

Why this matters:

- This is already very NIST-like. The parameters are being justified against the implementation that exists today, not against a simplified toy model.

### 4. Publish the exact estimator recipe

Question:

- If someone reruns the analysis, do they know exactly which model to use?

Current status: `partial`.

What is already good:

- `HACHI_SIS_ESTIMATION.md:64-78` explains that the note uses the actual Sage estimator, with `BDGL16` and `lgsa` forced explicitly for comparability.
- `HACHI_SIS_ESTIMATION.md:278-291` gives the exact `SIS.Parameters(...)` / `SIS.lattice(...)` shape.
- `src/protocol/labrador/config.rs:116-274` gives an in-tree mirror for the Euclidean path.

What is still missing:

- One canonical statement of the approved estimator matrix for Hachi parameter reviews:
  - Euclidean vs `l_infty`,
  - lattice-only vs other generic SIS attacks,
  - cost model choices,
  - shape model choices,
  - and which result is considered the security floor.

### 5. Check more than one attack family or cost view

Question:

- Are we taking the min across all credible public attacks, or just one convenient one?

Current status: `partial`.

What is already good:

- The current note is honest about the Euclidean path it uses and why.
- The broader investigation already distinguished Euclidean and `l_infty` views, and also looked at newer generic `l_infty` attacks.

What is still missing:

- A repo-local summary table that takes the minimum over:
  - the relevant commitment layers (`A`, `B/D`, `M`),
  - Euclidean and direct `l_infty` analyses,
  - and the best public generic attacks we currently know how to model.

Without that, the current numbers are strong evidence, but not yet a NIST-style final dossier.

### 6. Distinguish entropy floors from SIS floors

Question:

- Does a smaller `D` fail because binding gets weak, or because the challenge family stops being credible?

Current status: `good`.

What is already good:

- `src/protocol/commitment/config.rs:755-760` says the halving-D profile stops at `D = 128` because sparse ternary challenges still need enough entropy there.
- `HACHI_SIS_ESTIMATION.md:371-374` and `HACHI_SIS_ESTIMATION.md:476-483` make the same point explicitly: the current `D = 128 minimum` comment is a challenge-family statement, not an SIS statement.

Why this matters:

- This is exactly the kind of distinction that keeps parameter discussions honest. It prevents "the protocol fails here" from being shorthand for the wrong failure mode.

### 7. Include operational assumptions in the same document

Question:

- What system conditions have to hold for the parameter claim to mean anything?

Current status: `partial`.

What is already good:

- Transcript labels are centralized in `src/protocol/transcript/labels.rs:7-130`.
- Sparse challenge sampling explicitly absorbs the call-site label, instance index, ring degree, weight, and coefficient alphabet in `src/protocol/challenges/sparse.rs:36-46`.
- Proof payloads perform structural checks such as tail/config consistency and ring-dimension consistency in `src/protocol/proof.rs:914-942`.

What is still missing:

- One explicit checklist that says these are part of the security contract, alongside assumptions about:
  - Fiat-Shamir transcript collision resistance,
  - malformed proof rejection,
  - norm-bound enforcement,
  - and any side-channel or implementation assumptions that are relevant to the parameter claim.

### 8. Choose defaults with headroom, not merely feasibility

Question:

- Which parameter set would we recommend to someone who just wants a safe default?

Current status: `partial`.

What is already good:

- The current `Fp128` `D = 256` families are far above the 128-bit SIS floor under the current flattened Euclidean model.

What is still missing:

- A written rule saying which sets are comfortable defaults and which are only acceptable after additional review.

### 9. Define reevaluation triggers

Question:

- When do we revisit the parameter sheet?

Current status: `missing`.

Suggested triggers:

- any change to challenge family,
- any change to recursive reduction geometry,
- any change to the in-tree estimator formula,
- any new credible `l_infty` or module-aware SIS attack,
- or any reduction that materially changes the non-SIS soundness budget.

## Applied Readout For The Current Hachi SIS / MSIS Story

This section applies the checklist above to the current repo state and to `HACHI_SIS_ESTIMATION.md`.

### What already looks strong

1. The binding-side geometry is derived from live code, not from paper-only abstractions.
   `HACHI_SIS_ESTIMATION.md:76-78` and `src/protocol/labrador/config.rs:116-274` are the key anchors here.

2. The current repo already has a compact parameter menu instead of an unbounded tuning space.
   The important definitions are in `src/protocol/commitment/config.rs:679-814`.

3. The repo already distinguishes the main failure modes.
   The strongest example is the split between challenge entropy and SIS binding in `HACHI_SIS_ESTIMATION.md:652-664`.

4. Some operational security conditions are already encoded directly into the implementation.
   Transcript domain separation and proof-shape validation are not left entirely implicit; see
   `src/protocol/transcript/labels.rs:7-130`,
   `src/protocol/challenges/sparse.rs:36-46`,
   and `src/protocol/proof.rs:914-942`.

### Current binding-side numbers

Under the current conservative flattened Euclidean SIS model, the main current read is:

| Case | Weakest reported SIS number | Current interpretation |
| --- | ---: | --- |
| `Fp128Full`, `D = 256`, `max_num_vars = 25` | `725.67` bits | Very comfortable margin; weakest layer is `B/D` |
| `Fp128Full`, `D = 256`, `max_num_vars = 30` | `568.43` bits | Still nowhere near the 128-bit floor |
| `Fp128OneHot / LogBasis`, `D = 256`, `max_num_vars = 25` | `866.23` bits | Stronger than `Full` because `delta_commit` drops from `43` to `1` |
| `Fp128OneHot / LogBasis`, `D = 256`, `max_num_vars = 30` | `667.22` bits | Still very comfortable |
| Full-field ring sweep at `D = 128` | `315.58` bits | `D = 128` is not close to the SIS floor |
| Full-field ring sweep at `D = 64` with mixed challenge family | about `148.17 - 148.46` bits | Not immediately ruled out by SIS, but much closer to the floor |

Anchors:

- `HACHI_SIS_ESTIMATION.md:395-429`
- `HACHI_SIS_ESTIMATION.md:467-474`

The most important current conclusions remain:

- `HACHI_SIS_ESTIMATION.md:652-658` says the `D = 256` profiles are nowhere near the 128-bit SIS floor.
- `HACHI_SIS_ESTIMATION.md:655-658` says the current `D = 128 minimum` claim is really about the challenge family.
- `HACHI_SIS_ESTIMATION.md:657-658` says that a `D = 64` mixed-family shell still lands around `148` bits under the same conservative model.

That is already good evidence for internal tuning. But it is not yet the same thing as a NIST-style final parameter claim.

### Where the current dossier is still thin

1. The current repo-local writeup is strongest on the Euclidean SIS proxy.
   That is useful, but a NIST-style parameter sheet should also name the competing direct `l_infty` and other generic attack views that were checked and say which one wins.

2. There is not yet one end-to-end soundness budget table.
   The current SIS note is careful about binding, but the full Hachi claim also spends security on:
   - challenge entropy,
   - transcript-sampled reductions,
   - recursive aggregation soundness,
   - JL / Labrador terms if that tail is used,
   - and any proof-theoretic reduction losses.

3. There is not yet one default parameter recommendation with rationale.
   NIST-style practice would force a short statement such as:
   - "this is the default,"
   - "this is experimental but plausible,"
   - "this is research-only until more attack models are checked."

4. There is not yet one reevaluation policy.
   The repo has enough moving parts that this should be written down explicitly.

## Practical Recommendation

If we want a Hachi parameter workflow that really matches the NIST habit of mind, then for every proposed parameter set we should require one short dossier with the following contents:

1. The exact security claim being made.
2. The exact live-code geometry used to derive the SIS / MSIS instance.
3. The exact estimator versions and model choices.
4. The minimum security number across all relevant layers and all relevant attack views.
5. The challenge-entropy check, kept separate from the binding check.
6. The non-SIS soundness terms in the same table.
7. The operational assumptions and invalid-input / transcript conditions.
8. A statement saying whether the set is:
   - default,
   - experimental,
   - or research-only.

Applied to the current tree, my present read is:

- current `Fp128`, `D = 256` families look comfortably above the SIS floor,
- `D = 128` looks plausible from the SIS side and is blocked first by challenge-family choices rather than by binding,
- `D = 64` looks research-grade rather than default-grade,
- and the main thing missing is not more one-off SIS numbers, but one end-to-end security sheet that combines all the terms the way NIST would expect.

## One-Sentence Summary

The current Hachi SIS work is already strong on live-code geometry and on separating entropy from binding, but a fully NIST-style parameter claim still needs one consolidated, multi-model, end-to-end security dossier before a parameter family should be treated as a true default.
