use super::*;

use jolt_field::Zero;
use jolt_field::{One, Ring};

// ============================================================================
// GROUP E — Heterogeneous configurations (fp128)
//
// Tests that span multiple commitment groups with different polynomial types or
// compute backends.  Orthogonal to the Group B matrix.
// ============================================================================

// fp128: three commitment groups with heterogeneous polynomial types
// (one-hot precommit + dense precommit + one-hot final), proved jointly.
// This is the key test for the heterogeneous-group code path.
#[test]
fn heterogeneous_group_types() {
    init_rayon_pool();
    run_on_large_stack(|| {
        const ONEHOT_PRE_NV: usize = 14;
        const DENSE_PRE_NV: usize = 15;
        const FINAL_NV: usize = 16;
        let onehot_scheme =
            load_workspace_scheme::<OneHotCfg>().expect("workspace one-hot schedule catalog");
        let dense_scheme =
            load_workspace_scheme::<DenseCfg>().expect("workspace dense schedule catalog");

        let setup = onehot_scheme.setup_prover(FINAL_NV, 4).expect("setup");
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
        let stack =
            UniformProverStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
                .expect("stack");

        let onehot_k_pre =
            akita_config::unit_onehot_source_chunk_size::<OneHotCfg>().expect("one-hot config");
        let pre_chunks = (1usize << ONEHOT_PRE_NV) / onehot_k_pre;
        let onehot_pre = akita_prover::OneHotPoly::<F, u8>::new(
            onehot_k_pre,
            (0..pre_chunks)
                .map(|i| (i % 3 == 0).then_some((i % onehot_k_pre) as u8))
                .collect(),
        )
        .expect("K=256 precommitted poly");

        let dense_evals_a = (0..(1usize << DENSE_PRE_NV))
            .map(|i| F::from_u64((i % 257) as u64))
            .collect::<Vec<_>>();
        let dense_evals_b = (0..(1usize << DENSE_PRE_NV))
            .map(|i| F::from_u64((i % 509) as u64))
            .collect::<Vec<_>>();
        let dense_a = akita_prover::DensePoly::from_field_evals(DENSE_PRE_NV, &dense_evals_a)
            .expect("dense a");
        let dense_b = akita_prover::DensePoly::from_field_evals(DENSE_PRE_NV, &dense_evals_b)
            .expect("dense b");

        let final_onehot = make_onehot_poly::<OneHotCfg>(FINAL_NV, 0x1701_0000);

        let dense_polys = [dense_a.clone(), dense_b.clone()];
        let final_polys = [MultilinearPolynomial::onehot(final_onehot.clone())];

        // OneHot pre-group committed with OneHotCfg (matches catalog descriptor[0]).
        let akita_prover::CommitOutput {
            committed_group: onehot_pre_commitment,
            hint: onehot_pre_hint,
        } = onehot_scheme
            .commit(
                &setup,
                std::slice::from_ref(&onehot_pre),
                &stack,
                akita_prover::GroupContext::scheduler_without_precommitted_groups(),
            )
            .expect("K=256 precommit");

        // Dense pre-group committed with DenseCfg so its profile matches the
        // Dense descriptor in catalog entry {final_nv=16, pre=[onehot(14,1), dense(15,2)]}.
        let dense_setup = dense_scheme
            .setup_prover(DENSE_PRE_NV, 2)
            .expect("dense setup");
        let dense_prepared = CpuBackend::DEFAULT
            .prepare_setup(&dense_setup)
            .expect("dense prepared");
        let dense_stack = UniformProverStack::uniform(
            &CpuBackend::DEFAULT,
            &dense_prepared,
            dense_setup.expanded.as_ref(),
        )
        .expect("dense stack");
        let akita_prover::CommitOutput {
            committed_group: dense_commitment,
            hint: dense_hint,
        } = dense_scheme
            .commit(
                &dense_setup,
                &dense_polys,
                &dense_stack,
                akita_prover::GroupContext::scheduler_without_precommitted_groups(),
            )
            .expect("dense precommit");

        let precommitteds = PrecommittedGroupProfiles::from_profiles(vec![
            onehot_pre_commitment.profile,
            dense_commitment.profile,
        ])
        .expect("nonempty precommitted groups");
        let akita_prover::CommitOutput {
            committed_group: final_commitment,
            hint: final_hint,
        } = onehot_scheme
            .commit(
                &setup,
                &final_polys,
                &stack,
                akita_prover::GroupContext::scheduler_with_precommitted_groups(&precommitteds),
            )
            .expect("final commit");

        let onehot_pre_point: Vec<F> = (0..ONEHOT_PRE_NV)
            .map(|i| F::from_u64((i + 2) as u64))
            .collect();
        let dense_point: Vec<F> = (0..DENSE_PRE_NV)
            .map(|i| F::from_u64((i + 37) as u64))
            .collect();
        let final_point: Vec<F> = (0..FINAL_NV)
            .map(|i| F::from_u64((i + 71) as u64))
            .collect();

        // Independent oracles for every group in the heterogeneous batch.
        let onehot_pre_opening = onehot_opening_lagrange(&onehot_pre, &onehot_pre_point);
        let dense_opening_a = dense_opening_lagrange(&dense_evals_a, &dense_point);
        let dense_opening_b = dense_opening_lagrange(&dense_evals_b, &dense_point);
        let final_opening = onehot_opening_lagrange(&final_onehot, &final_point);

        let onehot_pre_refs = [&MultilinearPolynomial::onehot(onehot_pre.clone())];
        let dense_refs = [
            &MultilinearPolynomial::dense(dense_a.clone()),
            &MultilinearPolynomial::dense(dense_b.clone()),
        ];
        let final_refs = [&final_polys[0]];

        let prover_data = selected_prover_data::<OneHotCfg, _>(
            OpeningClaims::from_groups(vec![
                PolynomialGroupClaims::new(
                    onehot_pre_point.clone(),
                    vec![onehot_pre_opening],
                    onehot_pre_commitment.clone(),
                )
                .expect("K=256 prover group"),
                PolynomialGroupClaims::new(
                    dense_point.clone(),
                    vec![dense_opening_a, dense_opening_b],
                    dense_commitment.clone(),
                )
                .expect("dense prover group"),
                PolynomialGroupClaims::new(
                    final_point.clone(),
                    vec![final_opening],
                    final_commitment.clone(),
                )
                .expect("final prover group"),
            ])
            .expect("prover claims"),
            vec![onehot_pre_hint, dense_hint, final_hint],
            vec![&onehot_pre_refs, &dense_refs, &final_refs],
            onehot_scheme.schedules(),
        );
        let selection = prover_data.selection();

        // The openings below come from independent oracles, so the resolved
        // schedule is no longer needed to project them. Keep the resolution as
        // a structural check that the heterogeneous selection binds to the
        // two-precommit catalog entry.
        let schedule = onehot_scheme
            .schedules()
            .resolve_selection(selection)
            .expect("heterogeneous schedule")
            .schedule()
            .clone();
        assert_eq!(
            schedule.root.params.precommitted_groups().len(),
            2,
            "heterogeneous selection must resolve to the two-precommit entry"
        );

        let mut prover_transcript =
            AkitaTranscript::<F>::new(b"completeness/heterogeneous_group_types");
        let proof = onehot_scheme
            .batched_prove(
                &setup,
                prover_data,
                &stack,
                &mut prover_transcript,
                BasisMode::Lagrange,
            )
            .expect("heterogeneous prove");

        let shape = proof.shape();
        let mut bytes = Vec::new();
        proof.serialize_compressed(&mut bytes).expect("serialize");
        let decoded = AkitaBatchedProof::<F, F>::deserialize_compressed(
            &mut std::io::Cursor::new(bytes),
            &shape,
        )
        .expect("deserialize");

        let verifier_setup = onehot_scheme
            .setup_verifier(&setup)
            .expect("verifier setup");
        let verify_claims = OpeningClaims::from_groups(vec![
            PolynomialGroupClaims::new(
                onehot_pre_point,
                vec![onehot_pre_opening],
                &onehot_pre_commitment,
            )
            .expect("K=256 verifier group"),
            PolynomialGroupClaims::new(
                dense_point,
                vec![dense_opening_a, dense_opening_b],
                &dense_commitment,
            )
            .expect("dense verifier group"),
            PolynomialGroupClaims::new(final_point, vec![final_opening], &final_commitment)
                .expect("final verifier group"),
        ])
        .expect("verifier claims");
        let mut verifier_transcript =
            AkitaTranscript::<F>::new(b"completeness/heterogeneous_group_types");
        onehot_scheme
            .batched_verify(
                &decoded,
                &verifier_setup,
                &mut verifier_transcript,
                GroupBatchStatement::new(selection, verify_claims).expect("statement"),
                BasisMode::Lagrange,
            )
            .expect("heterogeneous verify");
    });
}

// fp128: a **bounded** dense precommit opened jointly with a one-hot final
// group. The two groups declare different committed-source bounds inside the same
// 128-bit field — `fp128::DenseBounded` commits against `log_commit_bound = 65`
// while the root is planned under `fp128::OneHot`'s `log_commit_bound = 1` — so
// this is the mixed-bound multi-group cell. It proves the bound is a per-group
// property frozen into each group's own A matrix, not a batch-wide one, and that a
// bounded group's full-width opening geometry still lines up with the shared root.
#[test]
fn bounded_dense_precommit_with_onehot_final_group() {
    type BoundedDenseCfg = fp128::DenseBounded;
    const BOUNDED_PRE_NV: usize = 14;
    const FINAL_NV: usize = 16;

    init_rayon_pool();
    run_on_large_stack(|| {
        let bounded_scheme = load_workspace_scheme::<BoundedDenseCfg>()
            .expect("workspace bounded-dense schedule catalog");
        let onehot_scheme =
            load_workspace_scheme::<OneHotCfg>().expect("workspace one-hot schedule catalog");
        // Full-width `u64` coefficients on both signs — the workload the bounded
        // preset exists for — including the `±u64::MAX` endpoints. `commit` must
        // accept all of them under `log_commit_bound = 65`, the signed bit width
        // whose range `[-2^64, 2^64 - 1]` contains every `u64`.
        let bounded_evals = u64_dense_field_evals(BOUNDED_PRE_NV, 0x8064_0001);
        let bounded_dense =
            akita_prover::DensePoly::from_field_evals(BOUNDED_PRE_NV, &bounded_evals)
                .expect("bounded dense poly");
        let final_onehot = make_onehot_poly::<OneHotCfg>(FINAL_NV, 0x8064_0000);

        // Each group commits under the config that owns its bound, so its frozen
        // profile matches the descriptor the catalog row carries.
        let bounded_setup = bounded_scheme
            .setup_prover(BOUNDED_PRE_NV, 1)
            .expect("bounded dense setup");
        let bounded_prepared = CpuBackend::DEFAULT
            .prepare_setup(&bounded_setup)
            .expect("bounded dense prepared");
        let bounded_stack = UniformProverStack::uniform(
            &CpuBackend::DEFAULT,
            &bounded_prepared,
            bounded_setup.expanded.as_ref(),
        )
        .expect("bounded dense stack");
        let akita_prover::CommitOutput {
            committed_group: bounded_commitment,
            hint: bounded_hint,
        } = bounded_scheme
            .commit(
                &bounded_setup,
                std::slice::from_ref(&bounded_dense),
                &bounded_stack,
                akita_prover::GroupContext::scheduler_without_precommitted_groups(),
            )
            .expect("bounded dense precommit");

        // The bounded group's source decomposition must be shallower than a
        // full-width decomposition at the same basis. The independently planned
        // full-width family may choose another basis, so comparing its raw digit
        // count would not compare like with like.
        let bounded_digits = bounded_commitment.profile.inner.digits;
        let full_width_digits_at_bounded_basis =
            akita_types::sis::compute_num_digits_field_width(128, bounded_digits.log_basis);
        assert!(
            bounded_digits.num_digits < full_width_digits_at_bounded_basis,
            "bounded precommit digit depth must be below same-basis full-width depth",
        );

        let setup = onehot_scheme
            .setup_prover(FINAL_NV, 2)
            .expect("one-hot root setup");
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
        let stack =
            UniformProverStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
                .expect("stack");

        let precommitteds =
            PrecommittedGroupProfiles::from_profiles(vec![bounded_commitment.profile])
                .expect("nonempty precommitted groups");
        let final_polys = [MultilinearPolynomial::onehot(final_onehot.clone())];
        let akita_prover::CommitOutput {
            committed_group: final_commitment,
            hint: final_hint,
        } = onehot_scheme
            .commit(
                &setup,
                &final_polys,
                &stack,
                akita_prover::GroupContext::scheduler_with_precommitted_groups(&precommitteds),
            )
            .expect("one-hot final commit against a bounded precommit");

        let bounded_point: Vec<F> = (0..BOUNDED_PRE_NV)
            .map(|i| F::from_u64((i + 11) as u64))
            .collect();
        let final_point: Vec<F> = (0..FINAL_NV)
            .map(|i| F::from_u64((i + 53) as u64))
            .collect();
        let bounded_opening = dense_opening_lagrange(&bounded_evals, &bounded_point);
        let final_opening = onehot_opening_lagrange(&final_onehot, &final_point);
        let bounded_point_for_tamper = bounded_point.clone();
        let final_point_for_tamper = final_point.clone();

        let bounded_refs = [&MultilinearPolynomial::dense(bounded_dense.clone())];
        let final_refs = [&final_polys[0]];

        let prover_data = selected_prover_data::<OneHotCfg, _>(
            OpeningClaims::from_groups(vec![
                PolynomialGroupClaims::new(
                    bounded_point.clone(),
                    vec![bounded_opening],
                    bounded_commitment.clone(),
                )
                .expect("bounded dense prover group"),
                PolynomialGroupClaims::new(
                    final_point.clone(),
                    vec![final_opening],
                    final_commitment.clone(),
                )
                .expect("final prover group"),
            ])
            .expect("prover claims"),
            vec![bounded_hint, final_hint],
            vec![&bounded_refs, &final_refs],
            onehot_scheme.schedules(),
        );
        let selection = prover_data.selection();

        let schedule = onehot_scheme
            .schedules()
            .resolve_selection(selection)
            .expect("mixed-bound schedule")
            .schedule()
            .clone();
        let precommitted = schedule
            .root
            .params
            .precommitted_groups()
            .first()
            .expect("mixed-bound selection must resolve the one-precommit entry");
        assert_eq!(schedule.root.params.precommitted_groups().len(), 1);
        assert_eq!(
            precommitted.profile.inner.digits.num_digits,
            bounded_commitment.profile.inner.digits.num_digits,
            "the resolved row must carry the bounded producer's own digit depth"
        );
        // The root itself is planned at the one-hot bound, so the two groups
        // really do disagree on their committed-source depth.
        assert_eq!(schedule.root.params.inner().digits.num_digits, 1,);

        let mut prover_transcript =
            AkitaTranscript::<F>::new(b"completeness/bounded_dense_precommit_with_onehot_final");
        let proof = onehot_scheme
            .batched_prove(
                &setup,
                prover_data,
                &stack,
                &mut prover_transcript,
                BasisMode::Lagrange,
            )
            .expect("mixed-bound prove");

        let shape = proof.shape();
        let mut bytes = Vec::new();
        proof.serialize_compressed(&mut bytes).expect("serialize");
        let decoded = AkitaBatchedProof::<F, F>::deserialize_compressed(
            &mut std::io::Cursor::new(bytes),
            &shape,
        )
        .expect("deserialize");

        let verifier_setup = onehot_scheme
            .setup_verifier(&setup)
            .expect("verifier setup");
        let verify_claims = OpeningClaims::from_groups(vec![
            PolynomialGroupClaims::new(bounded_point, vec![bounded_opening], &bounded_commitment)
                .expect("bounded dense verifier group"),
            PolynomialGroupClaims::new(final_point, vec![final_opening], &final_commitment)
                .expect("final verifier group"),
        ])
        .expect("verifier claims");
        let mut verifier_transcript =
            AkitaTranscript::<F>::new(b"completeness/bounded_dense_precommit_with_onehot_final");
        onehot_scheme
            .batched_verify(
                &decoded,
                &verifier_setup,
                &mut verifier_transcript,
                GroupBatchStatement::new(selection, verify_claims).expect("statement"),
                BasisMode::Lagrange,
            )
            .expect("mixed-bound verify");

        // Proves the verification above is load-bearing: the same proof against a
        // tampered bounded-group claim must be rejected.
        let tampered = OpeningClaims::from_groups(vec![
            PolynomialGroupClaims::new(
                bounded_point_for_tamper,
                vec![bounded_opening + F::one()],
                &bounded_commitment,
            )
            .expect("tampered bounded verifier group"),
            PolynomialGroupClaims::new(
                final_point_for_tamper,
                vec![final_opening],
                &final_commitment,
            )
            .expect("final verifier group"),
        ])
        .expect("tampered claims");
        let mut tampered_transcript =
            AkitaTranscript::<F>::new(b"completeness/bounded_dense_precommit_with_onehot_final");
        assert!(
            onehot_scheme
                .batched_verify(
                    &decoded,
                    &verifier_setup,
                    &mut tampered_transcript,
                    GroupBatchStatement::new(selection, tampered).expect("statement"),
                    BasisMode::Lagrange,
                )
                .is_err(),
            "a tampered bounded-group opening must not verify"
        );
    });
}

// The bounded family's own scalar rows: a direct commit/prove/verify round trip
// at every `nv` its catalog ships, over the workload the preset exists for —
// full-width `u64` coefficients on both signs, including the `±u64::MAX`
// endpoints.
//
// This is the test that pins the preset against its *purpose* rather than
// against its declaration. A generator narrowed to fit the declared bound would
// pass even if the bound were declared too narrow, which is exactly how a
// signed/unsigned mix-up ships silently.
#[test]
fn bounded_dense_roundtrip_over_u64_coefficients_at_every_catalog_size() {
    init_rayon_pool();
    run_on_large_stack(|| {
        prove_verify_dense_roundtrip_with_evals::<fp128::DenseBounded>(
            &[14, 24, 26],
            b"completeness/fp128_dense_bounded",
            u64_dense_field_evals,
        );
    });
}

// A schedule's declared source *class* is a producer obligation, not only a
// planning input, and it is independent of the numeric bound.
//
// `fp128::OneHot` prices at most one hot position per 256 source coefficients.
// A dense polynomial whose values are all `1` reports centered reach `(0, 1)`,
// which sits comfortably inside that schedule's one-digit `[-4, 3]` balanced
// envelope — so every magnitude test admits it, while it carries up to 256x the
// per-chunk energy the frozen response caps were planned for. Only a check on the
// representation itself can reject it, and it must be rejected at `commit` rather
// than surface later as an unexplained proof failure.
//
// The class must come from the declared honest-fold policy, never from
// `log_commit_bound == 1`: this family deliberately allows class and bound to vary
// independently, so inferring one from the other is the bug this guards.
#[test]
fn commit_rejects_a_source_whose_representation_is_not_the_declared_class() {
    const NV: usize = 14;

    init_rayon_pool();
    run_on_large_stack(|| {
        let scheme = load_workspace_scheme::<OneHotCfg>().expect("workspace schedule catalog");
        let setup = scheme.setup_prover(NV, 1).expect("setup");
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
        let stack =
            UniformProverStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
                .expect("stack");

        let profile = scheme
            .schedules()
            .resolve_key(&akita_types::AkitaScheduleLookupKey::single(
                akita_types::PolynomialGroupLayout::new(NV, 1),
            ))
            .expect("one-hot row")
            .profiles()
            .final_group;
        // Dense all-ones: inside the digit envelope, outside the source class.
        let dense = akita_prover::DensePoly::<F>::from_field_evals(NV, &[F::one(); 1usize << NV])
            .expect("dense poly");
        let error = scheme
            .commit(
                &setup,
                std::slice::from_ref(&dense),
                &stack,
                akita_prover::GroupContext::scheduler_without_precommitted_groups(),
            )
            .map(|_| ())
            .expect_err("a dense source must not commit under a one-hot schedule");
        assert!(
            matches!(error, akita_error::AkitaError::InvalidInput(_)),
            "expected InvalidInput, got {error:?}"
        );

        // The magnitudes really were admissible, so the rejection is the class
        // check doing work rather than the interval check catching it anyway.
        let contract = OneHotCfg::committed_source_contract()
            .expect("the one-hot preset declares a valid producer contract");
        let (negative, positive) = contract.accepted_bounds(
            profile.inner.digits.log_basis,
            profile.inner.digits.num_digits,
        );
        assert!(
            negative.is_some_and(|reach| reach >= 1) && positive.is_some_and(|reach| reach >= 1),
            "the all-ones dense source fits the accepted magnitude interval [{negative:?}, {positive:?}]"
        );

        // The proper one-hot representation at the same geometry is accepted.
        let onehot_k =
            akita_config::unit_onehot_source_chunk_size::<OneHotCfg>().expect("one-hot config");
        let onehot = akita_prover::OneHotPoly::<F, u8>::new(
            onehot_k,
            (0..(1usize << NV) / onehot_k)
                .map(|i| Some((i % onehot_k) as u8))
                .collect(),
        )
        .expect("one-hot poly");
        scheme
            .commit(
                &setup,
                std::slice::from_ref(&onehot),
                &stack,
                akita_prover::GroupContext::scheduler_without_precommitted_groups(),
            )
            .expect("the declared one-hot representation must still commit");
    });
}

/// The producer contract `fp128::DenseBounded` declares: class plus bound, read
/// from the config rather than restated, so these tests cannot drift from it.
fn bounded_contract() -> akita_types::sis::CommittedSourceContract {
    fp128::DenseBounded::committed_source_contract()
        .expect("the bounded preset declares a valid producer contract")
}

// The declared bound must contain every `u64`, and must say so in the type's own
// terms rather than only through the digit geometry.
#[test]
fn bounded_dense_declares_a_bound_that_contains_every_u64() {
    type BoundedDenseCfg = fp128::DenseBounded;

    // `log_commit_bound` is a *signed* bit width: `k` is `[-2^(k-1), 2^(k-1) - 1]`.
    // Covering `u64::MAX = 2^64 - 1` therefore takes 65, not 64.
    assert_eq!(BoundedDenseCfg::LOG_COMMIT_BOUND, 65);
    const { assert!(BoundedDenseCfg::MAX_CENTERED_MAGNITUDE >= u64::MAX as u128) };
    assert_eq!(
        BoundedDenseCfg::MAX_CENTERED_MAGNITUDE,
        u128::from(u64::MAX),
        "the positive endpoint must be exactly u64::MAX"
    );

    // The declared interval, read from the config's own producer contract.
    let (negative, positive) = bounded_contract().declared_bounds();
    assert_eq!(positive, Some(u128::from(u64::MAX)));
    assert_eq!(negative, Some(1u128 << 64));

    // And a 64-bit *signed* declaration would not have covered it — the
    // off-by-one this guards against.
    let (_, signed_64_positive) = akita_types::sis::CommittedSourceContract::try_new(
        BoundedDenseCfg::committed_source_class(),
        akita_types::DecompositionParams {
            log_commit_bound: 64,
            ..BoundedDenseCfg::decomposition()
        },
    )
    .expect("a signed 64-bit declaration is representable, just too narrow")
    .declared_bounds();
    assert!(
        signed_64_positive.expect("interior bound") < u128::from(u64::MAX),
        "a signed 64-bit bound must not reach u64::MAX; that is why the preset declares 65"
    );
}

// A polynomial outside the *declared* bound must be rejected at commit time.
//
// Two independent reasons, and the guard has to enforce the tighter one:
//
//  * Representability — the decomposition kernel peels exactly
//    `num_digits_inner` digits and discards the rest, so a commitment would
//    silently bind a truncation.
//  * Declaration — the planner prices a bounded source's final digit plane at
//    only the range its bound leaves, so a coefficient past the declaration
//    inflates the level-1 witness beyond the frozen L2 response caps.
//
// The digit envelope is far wider than the declaration (the depth rounds up), so
// checking representability alone would accept coefficients the schedule was
// never priced for. This test pins the *declaration* as the accepted interval.
#[test]
fn bounded_dense_commit_rejects_a_coefficient_above_the_declared_bound() {
    type BoundedDenseCfg = fp128::DenseBounded;
    const NV: usize = 14;

    init_rayon_pool();
    run_on_large_stack(|| {
        let bounded_scheme = load_workspace_scheme::<BoundedDenseCfg>()
            .expect("workspace bounded-dense schedule catalog");
        let dense_scheme =
            load_workspace_scheme::<DenseCfg>().expect("workspace dense schedule catalog");
        let setup = bounded_scheme.setup_prover(NV, 1).expect("bounded setup");
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
        let stack =
            UniformProverStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
                .expect("stack");

        let commit = |evals: &[F]| {
            let poly = akita_prover::DensePoly::from_field_evals(NV, evals).expect("dense poly");
            bounded_scheme
                .commit(
                    &setup,
                    std::slice::from_ref(&poly),
                    &stack,
                    akita_prover::GroupContext::scheduler_without_precommitted_groups(),
                )
                .map(|_| ())
        };

        // The accepted interval, read from the config rather than recomputed, so
        // the test cannot drift from the shipped declaration.
        let (negative_bound, positive_bound) = bounded_contract().declared_bounds();
        let negative_bound = negative_bound.expect("an interior bound is constrained");
        let positive_bound = positive_bound.expect("an interior bound is constrained");

        // Both signed endpoints of the declaration, and every `u64` in between.
        let mut in_bound = vec![F::zero(); 1usize << NV];
        in_bound[0] = F::from_u128(positive_bound);
        in_bound[1] = -F::from_u128(negative_bound);
        for (slot, value) in u64_magnitude_endpoints().into_iter().enumerate() {
            in_bound[2 + slot] = value;
        }
        commit(&in_bound).expect("both declared endpoints and every u64 must be accepted");

        // One past either endpoint is an input error.
        let mut over_positive = vec![F::zero(); 1usize << NV];
        over_positive[3] = F::from_u128(positive_bound + 1);
        let error = commit(&over_positive)
            .expect_err("a coefficient above the declared bound must be rejected");
        assert!(
            matches!(error, akita_error::AkitaError::InvalidInput(_)),
            "expected InvalidInput, got {error:?}"
        );

        let mut over_negative = vec![F::zero(); 1usize << NV];
        over_negative[3] = -F::from_u128(negative_bound + 1);
        assert!(
            commit(&over_negative).is_err(),
            "a coefficient below the declared bound must be rejected"
        );

        // The rejected value is still well inside what the digits can *represent*.
        // Without the declaration half of the accepted-interval intersection this
        // would commit successfully on a schedule priced for a narrower range —
        // the regression this test exists to catch.
        let profile = bounded_scheme
            .schedules()
            .resolve_key(&akita_types::AkitaScheduleLookupKey::single(
                akita_types::PolynomialGroupLayout::new(NV, 1),
            ))
            .expect("bounded row")
            .profiles()
            .final_group;
        let (_, representable) = akita_types::sis::checked_balanced_digit_representable_bounds(
            profile.inner.digits.log_basis,
            profile.inner.digits.num_digits,
        );
        assert!(
            representable.expect("shipped geometry fits u128") > positive_bound,
            "the digit envelope must be strictly wider than the declaration, \
             otherwise this test proves nothing about which one is enforced"
        );

        // A full-width dense commitment of the same out-of-range polynomial is
        // fine: it declares the whole field, so the guard is specific to a bounded
        // source and costs unbounded configs nothing.
        let full_setup = dense_scheme.setup_prover(NV, 1).expect("setup");
        let full_prepared = CpuBackend::DEFAULT
            .prepare_setup(&full_setup)
            .expect("prepared");
        let full_stack = UniformProverStack::uniform(
            &CpuBackend::DEFAULT,
            &full_prepared,
            full_setup.expanded.as_ref(),
        )
        .expect("stack");
        let poly =
            akita_prover::DensePoly::from_field_evals(NV, &over_positive).expect("dense poly");
        dense_scheme
            .commit(
                &full_setup,
                std::slice::from_ref(&poly),
                &full_stack,
                akita_prover::GroupContext::scheduler_without_precommitted_groups(),
            )
            .expect("full-width dense accepts every field element");
    });
}

// Compute backend heterogeneity: commit uses CpuBackend, prove uses a split
// ProverComputeStack with separate backends for each phase.
#[test]
fn heterogeneous_compute_backends() {
    init_rayon_pool();
    run_on_large_stack(|| {
        const NV: usize = 16;
        type Cfg = fp128::Dense;
        let scheme = load_workspace_scheme::<Cfg>().expect("workspace schedule catalog");

        let evals: Vec<F> = (0..(1usize << NV)).map(|i| F::from_u64(i as u64)).collect();
        let poly = akita_prover::DensePoly::<F>::from_field_evals(NV, &evals).unwrap();

        let setup = scheme.setup_prover(NV, 1).unwrap();
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");

        let commit_backend = CommitCluster;
        let opening_backend = OpeningCluster;
        let tensor = TensorCluster;
        let ring = RingSwitchCluster;
        let stack = ProverComputeStack::new(
            (&commit_backend, &prepared),
            (&opening_backend, &prepared),
            (&tensor, &prepared),
            (&ring, &prepared),
            setup.expanded.as_ref(),
        )
        .expect("heterogeneous stack");

        let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");
        let commit_stack =
            UniformProverStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
                .expect("commit stack");
        let akita_prover::CommitOutput {
            committed_group: commitment,
            hint,
        } = akita_prover::commit::<Cfg, akita_prover::DensePoly<F>, CpuBackend>(
            std::slice::from_ref(&poly),
            setup.expanded.as_ref(),
            scheme.schedules(),
            &commit_stack,
            akita_prover::GroupContext::scheduler_without_precommitted_groups(),
        )
        .expect("commit");

        let pt: Vec<F> = (0..NV).map(|i| F::from_u64((i + 2) as u64)).collect();
        let expected_opening = dense_opening_lagrange(&evals, &pt);

        let poly_refs = [&poly];
        let commitments = [commitment];
        let prover_data = selected_prover_data::<Cfg, _>(
            OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                pt.clone(),
                vec![expected_opening],
                commitments[0].clone(),
            )
            .expect("prover group")])
            .expect("prover claims"),
            vec![hint],
            vec![&poly_refs[..]],
            scheme.schedules(),
        );
        let selection = prover_data.selection();

        let mut prover_transcript =
            AkitaTranscript::<F>::new(b"completeness/heterogeneous_compute_backends");
        let proof = batched_prove::<Cfg, _, _, _, _, _, _>(
            &setup.expanded,
            &setup.prefix_slots,
            scheme.schedules(),
            &stack,
            prover_data,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("heterogeneous prove");

        let shape = proof.shape();
        let mut bytes = Vec::new();
        proof.serialize_compressed(&mut bytes).expect("serialize");
        let decoded = AkitaBatchedProof::<F, F>::deserialize_compressed(
            &mut std::io::Cursor::new(bytes),
            &shape,
        )
        .expect("deserialize");

        let mut verifier_transcript =
            AkitaTranscript::<F>::new(b"completeness/heterogeneous_compute_backends");
        scheme
            .batched_verify(
                &decoded,
                &verifier_setup,
                &mut verifier_transcript,
                GroupBatchStatement::new(
                    selection,
                    OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                        pt.clone(),
                        vec![expected_opening],
                        &commitments[0],
                    )
                    .expect("verifier group")])
                    .expect("verifier claims"),
                )
                .expect("statement"),
                BasisMode::Lagrange,
            )
            .expect("heterogeneous verify");
    });
}
