use super::*;
use akita_field::RandomSampling;
use akita_field::{FpExt4, Prime24Offset3};
use akita_types::tensor_equality_factor_evals;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

type F = Prime24Offset3;
type E = FpExt4<F>;

/// One entry per stride-window at a random within-window offset — the real
/// `np = 1` EOR witness shape (`stride = onehot_k / width = 2^s`).
fn build_np1_witness<B, G>(
    log_chunks: usize,
    s: usize,
    rng: &mut StdRng,
) -> SparseExtensionOpeningWitness<B, G>
where
    B: FieldCore,
    G: ExtField<B> + RandomSampling,
{
    let stride = 1usize << s;
    let num_chunks = 1usize << log_chunks;
    let table_len = num_chunks * stride;
    let mut entries = Vec::with_capacity(num_chunks);
    for chunk in 0..num_chunks {
        let off = (rng.next_u32() as usize) & (stride - 1);
        entries.push((chunk * stride + off, G::random(rng)));
    }
    SparseExtensionOpeningWitness::new(table_len, entries).unwrap()
}

/// Standard multilinear fold of a dense factor table: `A'[p] = (1-r)·A[2p] +
/// r·A[2p+1]`. Keeps the factor in lock-step with the folding witness.
fn fold_dense<G: FieldCore>(factor: &mut Vec<G>, r: G) {
    let one_minus = G::one() - r;
    let half = factor.len() / 2;
    for p in 0..half {
        factor[p] = factor[2 * p] * one_minus + factor[2 * p + 1] * r;
    }
    factor.truncate(half);
}

/// The closed-form merge-free path must produce byte-identical round messages
/// and byte-identical folded entries vs the general grouped/realloc path, in
/// both the sequential and the parallel (`>= SPARSE_PARALLEL_ENTRY_THRESHOLD`,
/// 2^14 entries) regimes.
#[test]
fn merge_free_matches_general_round_by_round() {
    for (log_chunks, s) in [(8usize, 4usize), (14usize, 4usize)] {
        let mut rng = StdRng::seed_from_u64(0x1234_5678 ^ ((log_chunks as u64) << 8));
        let coeff = E::random(&mut rng);

        let mut fast = build_np1_witness::<F, E>(log_chunks, s, &mut rng);
        // Reference: the identical witness with the fast path disabled.
        let mut reference = fast.clone();
        reference.merge_free_rounds_left = 0;

        // The guaranteed plateau is exactly log2(stride) = s, independent of
        // the random within-window offsets.
        assert_eq!(fast.merge_free_rounds_left, s, "unexpected plateau length");

        let mut factor: Vec<E> = (0..fast.table_len()).map(|_| E::random(&mut rng)).collect();

        let rounds = fast.table_len().trailing_zeros() as usize;
        for round in 0..rounds {
            let (mut c_fast, mut q_fast) = (E::zero(), E::zero());
            fast.accumulate_round(&factor, coeff, &mut c_fast, &mut q_fast);
            let (mut c_ref, mut q_ref) = (E::zero(), E::zero());
            reference.accumulate_round(&factor, coeff, &mut c_ref, &mut q_ref);
            assert_eq!(
                c_fast, c_ref,
                "constant mismatch (log_chunks={log_chunks}, round={round})"
            );
            assert_eq!(
                q_fast, q_ref,
                "quadratic mismatch (log_chunks={log_chunks}, round={round})"
            );

            if round < s {
                assert!(
                    fast.merge_free_rounds_left > 0,
                    "fast path disengaged during plateau (round={round})"
                );
            }

            let r = E::random(&mut rng);
            fast.fold_in_place(r);
            reference.fold_in_place(r);
            assert_eq!(fast.table_len(), reference.table_len());
            assert_eq!(fast.indices(), reference.indices());
            for row in 0..fast.num_entries() {
                assert_eq!(
                    fast.value(row),
                    reference.value(row),
                    "folded value mismatch (log_chunks={log_chunks}, round={round}, row={row})"
                );
            }
            fold_dense(&mut factor, r);
        }
        assert_eq!(fast.table_len(), 1);
        assert_eq!(fast.merge_free_rounds_left, 0);
    }
}

/// The production term path (sparse fused fold+accumulate inside the plateau,
/// general path afterwards) must emit byte-identical per-round messages and a
/// byte-identical final `(witness, factor)` vs a reference that runs the
/// original general accumulate + general fold every round (fusion AND the
/// merge-free fast path both disabled). Covers the sequential and parallel
/// (`>= 2^14` entries) regimes.
///
/// Uses `FpExt4<Fp32>` (the production fp32 `onehot_d32` field): it
/// implements `HasOptimizedFold`, required by the term-level fused path, and
/// takes the delayed-reduction accumulator (`DELAYED_PRODUCT_SUM_IS_EXACT`).
#[test]
fn fused_term_matches_unfused_reference() {
    use akita_field::{FpExt4, Prime32Offset99};
    type TE = FpExt4<Prime32Offset99>;
    let plan = SumcheckKernelPlan::detect();

    for (log_chunks, s) in [(6usize, 4usize), (14usize, 4usize)] {
        let mut rng = StdRng::seed_from_u64(0xfeed_1234 ^ ((log_chunks as u64) << 8));
        let coeff = TE::random(&mut rng);

        let witness = build_np1_witness::<Prime32Offset99, TE>(log_chunks, s, &mut rng);
        let table_len = witness.table_len();
        assert_eq!(
            witness.merge_free_rounds_left, s,
            "unexpected plateau length"
        );
        let factor: Vec<TE> = (0..table_len).map(|_| TE::random(&mut rng)).collect();

        // Fused: the production term path engages the sparse fused fold.
        let mut fused_term = ExtensionOpeningReductionTerm::<Prime32Offset99, TE>::new_sparse(
            witness.clone(),
            factor.clone(),
            coeff,
        )
        .unwrap();

        // Reference: identical inputs, but merge-free disabled so every round
        // takes the original general accumulate + general fold (no fusion).
        let mut ref_witness = witness;
        ref_witness.merge_free_rounds_left = 0;
        let mut ref_factor = factor;

        let rounds = table_len.trailing_zeros() as usize;
        let challenges: Vec<TE> = (0..rounds).map(|_| TE::random(&mut rng)).collect();

        for (round, &r) in challenges.iter().enumerate() {
            let (mut c_fused, mut q_fused) = (TE::zero(), TE::zero());
            fused_term.accumulate_into(plan, &mut c_fused, &mut q_fused);

            let (mut c_ref, mut q_ref) = (TE::zero(), TE::zero());
            ref_witness.accumulate_round(&ref_factor, coeff, &mut c_ref, &mut q_ref);

            assert_eq!(
                c_fused, c_ref,
                "constant mismatch (log_chunks={log_chunks}, round={round})"
            );
            assert_eq!(
                q_fused, q_ref,
                "quadratic mismatch (log_chunks={log_chunks}, round={round})"
            );

            fused_term.ingest_challenge(plan, r);
            ref_witness.fold_in_place(r);
            fold_dense(&mut ref_factor, r);
        }

        let (w_fused, f_fused) = fused_term
            .final_witness_and_factor_evals()
            .expect("fused term fully folded");
        assert_eq!(
            w_fused,
            ref_witness.final_eval().unwrap(),
            "final witness mismatch (log_chunks={log_chunks})"
        );
        assert_eq!(
            f_fused, ref_factor[0],
            "final factor mismatch (log_chunks={log_chunks})"
        );
    }
}

#[test]
fn grouped_tensor_fold_respects_merge_free_boundary() {
    use akita_field::{FpExt4, Prime32Offset99};
    type B = Prime32Offset99;
    type G = FpExt4<B>;

    let mut rng = StdRng::seed_from_u64(0x7e11_50a7);
    let palette = (0..4).map(|_| G::random(&mut rng)).collect::<Vec<_>>();
    let witness = SparseExtensionOpeningWitness::new(
        1 << 9,
        (0..64)
            .map(|chunk| {
                let offset = (rng.next_u32() as usize) & 7;
                (chunk * 8 + offset, palette[chunk & 3])
            })
            .collect(),
    )
    .unwrap();
    assert_eq!(witness.merge_free_rounds_left, 3);
    let tail_point = (0..9).map(|_| G::random(&mut rng)).collect::<Vec<_>>();
    let eta = (0..2).map(|_| G::random(&mut rng)).collect::<Vec<_>>();
    let factor = tensor_equality_factor_evals::<B, G>(&tail_point, &eta).unwrap();
    let coeff = G::random(&mut rng);

    let mut grouped = ExtensionOpeningReductionTerm::new_sparse_tensor_factor(
        witness.clone(),
        tail_point,
        eta,
        coeff,
        5,
    )
    .unwrap();
    let mut dense =
        ExtensionOpeningReductionTerm::new_sparse(witness.clone(), factor, coeff).unwrap();
    let plan = SumcheckKernelPlan::detect();

    for round in 0..9 {
        let mut grouped_round = (G::zero(), G::zero());
        grouped.accumulate_into(plan, &mut grouped_round.0, &mut grouped_round.1);
        let mut dense_round = (G::zero(), G::zero());
        dense.accumulate_into(plan, &mut dense_round.0, &mut dense_round.1);
        assert_eq!(grouped_round, dense_round, "round {round}");

        let challenge = G::random(&mut rng);
        grouped.ingest_challenge(plan, challenge);
        dense.ingest_challenge(plan, challenge);
    }
    assert_eq!(
        grouped.final_witness_and_factor_evals(),
        dense.final_witness_and_factor_evals()
    );
}

#[test]
fn cylindrical_term_matches_materialized_high_variable_extension() {
    use akita_field::{FpExt4, Prime32Offset99};
    type TE = FpExt4<Prime32Offset99>;
    let plan = SumcheckKernelPlan::detect();

    let mut rng = StdRng::seed_from_u64(0xc711_1d3a);
    let native_witness = (0..8).map(|_| TE::random(&mut rng)).collect::<Vec<_>>();
    let native_factor = (0..8).map(|_| TE::random(&mut rng)).collect::<Vec<_>>();
    let coeff = TE::random(&mut rng);

    let mut cylindrical = ExtensionOpeningReductionTerm::<Prime32Offset99, TE>::new(
        native_witness.clone(),
        native_factor.clone(),
        coeff,
    )
    .expect("native term")
    .extend_cylindrically(vec![TE::zero(), TE::zero()])
    .expect("virtual extension");

    let mut materialized_witness = Vec::with_capacity(32);
    for _ in 0..4 {
        materialized_witness.extend_from_slice(&native_witness);
    }
    let mut materialized_factor = native_factor;
    materialized_factor.resize(32, TE::zero());
    let mut materialized = ExtensionOpeningReductionTerm::<Prime32Offset99, TE>::new(
        materialized_witness,
        materialized_factor,
        coeff,
    )
    .expect("materialized extension");

    for round in 0..5 {
        let (mut cylindrical_constant, mut cylindrical_quadratic) = (TE::zero(), TE::zero());
        cylindrical.accumulate_into(plan, &mut cylindrical_constant, &mut cylindrical_quadratic);
        let (mut materialized_constant, mut materialized_quadratic) = (TE::zero(), TE::zero());
        materialized.accumulate_into(
            plan,
            &mut materialized_constant,
            &mut materialized_quadratic,
        );
        assert_eq!(
            (cylindrical_constant, cylindrical_quadratic),
            (materialized_constant, materialized_quadratic),
            "round {round}"
        );
        let challenge = TE::random(&mut rng);
        cylindrical.ingest_challenge(plan, challenge);
        materialized.ingest_challenge(plan, challenge);
    }
    assert_eq!(
        cylindrical.final_witness_and_factor_evals(),
        materialized.final_witness_and_factor_evals()
    );
}
