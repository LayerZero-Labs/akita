use super::*;
use akita_algebra::offset_eq::eq_eval_at_index;
use jolt_algebra::{Prime128Offset275 as F, Ring, Zero};
use rand::{rngs::StdRng, Rng, SeedableRng};

#[test]
fn paired_offset_equality_sums_match_direct_summation() {
    let mut rng = StdRng::seed_from_u64(7);
    for _ in 0..300 {
        let a = (0..rng.gen_range(0..9))
            .map(|_| F::from_u64(rng.gen()))
            .collect::<Vec<_>>();
        let b = (0..rng.gen_range(0..9))
            .map(|_| F::from_u64(rng.gen()))
            .collect::<Vec<_>>();
        let (u, v, count) = (
            rng.gen_range(0..300),
            rng.gen_range(0..300),
            rng.gen_range(0..300),
        );
        let direct = (0..count).fold(F::zero(), |sum, q| {
            sum + eq_eval_at_index(&a, u + q) * eq_eval_at_index(&b, v + q)
        });
        assert_eq!(
            paired_offset_eq_sum(&a, u, &b, v, count).unwrap(),
            direct,
            "{u} {v} {count}"
        );
    }
}

#[test]
fn p64_typed_prefix_matches_base_setup_and_extension_authentication() {
    use crate::binary_source::{binary_digit_authentication_weights, binary_word_manifest_fixture};
    use akita_params::binary_setup::BinaryFieldProfile;
    use jolt_algebra::{Ext2, ExtField, Prime64Offset59 as Base};
    type E = Ext2<Base>;
    for (word, planes) in [(true, 0), (false, 2), (true, 2)] {
        let manifest = binary_word_manifest_fixture(BinaryFieldProfile::P64, 1);
        let plan = BinaryRootWitnessPlan::for_profile(
            manifest.descriptor(),
            3,
            64,
            BinaryFieldProfile::P64,
            word,
            planes,
        )
        .unwrap();
        let typed = BinaryTypedPrefix::new(&manifest, word, planes, 64).unwrap();
        assert_eq!(
            typed,
            BinaryTypedPrefix::from_plan(&manifest, &plan).unwrap()
        );
        assert_eq!(
            typed.rows,
            [
                manifest.b_rows(),
                manifest.d_label(word, planes).unwrap().rank as usize
            ]
        );
        if planes == 0 {
            assert_ne!(typed.rows[0], typed.rows[1]);
        }
        let extension =
            |i: usize| E::from_base_fn(|c| Base::from_u64((3 + i * 17 + c * 29) as u64));
        let root_point = (0..plan.repacking().domain_len().trailing_zeros())
            .map(|i| extension(i as usize))
            .collect::<Vec<_>>();
        let rho = (0..3)
            .map(|r| std::array::from_fn(|k| extension(r * 648 + k)))
            .collect::<Vec<_>>();
        let a = BinaryAProduct::new(&plan, &rho, &root_point, typed.domain_len).unwrap();
        let batches = typed
            .rows
            .map(|rows| (0..rows).map(|r| extension(3000 + r)).collect::<Vec<_>>());
        let refs = [&batches[0][..], &batches[1][..]];
        let seed = [41; 32];
        let base = typed
            .setup_coefficients::<Base>(&seed, &manifest, typed.domain_len)
            .unwrap();
        let extension_setup = typed
            .setup_coefficients::<E>(&seed, &manifest, typed.domain_len)
            .unwrap();
        assert_eq!(
            extension_setup,
            base.iter().copied().map(E::lift_base).collect::<Vec<_>>()
        );
        let alpha = extension(8000);
        let dense = typed
            .prover_weights(&a, &root_point, alpha, refs, typed.domain_len)
            .unwrap();
        let mut blocks = typed
            .prover_weight_blocks(&a, &root_point, alpha, refs, typed.domain_len)
            .unwrap();
        let mut expanded = vec![E::zero(); typed.domain_len];
        for (live, &block) in blocks.blocks().iter().enumerate() {
            for (scale, row) in blocks.block_terms(live).unwrap() {
                for (slot, &value) in row.iter().enumerate() {
                    expanded[64 * block + slot] += scale * value;
                }
            }
        }
        assert_eq!(dense, expanded);
        let point = (0..typed.domain_len.trailing_zeros())
            .map(|i| extension(9000 + i as usize))
            .collect::<Vec<_>>();
        let expected = dense
            .iter()
            .zip(EqPolynomial::evals(&point).unwrap())
            .fold(E::zero(), |sum, (&value, eq)| sum + value * eq);
        assert_eq!(
            expected,
            a.evaluate(&point).unwrap()
                + typed
                    .digit_weight(&point, &root_point, alpha, refs)
                    .unwrap()
        );
        let matrix = binary_root_matrix(&seed, &manifest).unwrap();
        let a_setup =
            binary_a_setup_coefficients::<Base>(&matrix, &manifest, typed.domain_len).unwrap();
        let (index, coefficient) = a.factors().unwrap();
        let a_claim = a_setup
            .iter()
            .enumerate()
            .fold(E::zero(), |sum, (cell, &value)| {
                sum + E::lift_base(value)
                    * index[cell / coefficient.len()]
                    * coefficient[cell % coefficient.len()]
            });
        let mut direct = a_claim;
        for (role, segments) in [vec![1], plan.opening_segments()].iter().enumerate() {
            let weights = binary_digit_authentication_weights(
                &plan,
                segments,
                &seed,
                typed.labels[role],
                &vec![E::zero(); typed.rows[role] * 64],
                alpha,
                refs[role],
            )
            .unwrap();
            direct += weights.evaluate(&root_point).unwrap();
        }
        let typed_claim = base
            .iter()
            .zip(&dense)
            .fold(E::zero(), |sum, (&setup, &weight)| {
                sum + E::lift_base(setup) * weight
            });
        assert_eq!(typed_claim, direct);
        assert_ne!(typed_claim.base_coefficient(1), Base::zero());
        for bit in 0..6 {
            blocks.bind(point[bit]).unwrap();
            for (live, &block) in blocks.blocks().iter().enumerate() {
                let terms = blocks.block_terms(live).unwrap();
                for slot in 0..blocks.width() {
                    let first = 64 * block + slot * (1 << (bit + 1));
                    let eq = EqPolynomial::evals(&point[..=bit]).unwrap();
                    let expected = dense[first..first + eq.len()]
                        .iter()
                        .zip(eq)
                        .fold(E::zero(), |sum, (&v, e)| sum + v * e);
                    assert_eq!(
                        terms
                            .iter()
                            .fold(E::zero(), |sum, (scale, row)| sum + *scale * row[slot]),
                        expected
                    );
                }
            }
        }
        if typed.rows[0] != typed.rows[1] {
            assert!(typed
                .prover_weight_blocks(
                    &a,
                    &root_point,
                    alpha,
                    [&batches[0], &batches[0]],
                    typed.domain_len,
                )
                .is_err());
        }
        assert!(typed
            .prover_weight_blocks(&a, &root_point, alpha, [&batches[0], &[]], typed.domain_len)
            .is_err());
    }
}

#[test]
fn p64_typed_layout_uses_separate_admitted_ranks_without_large_witnesses() {
    use crate::binary_source::binary_word_manifest_fixture;
    use akita_params::binary_setup::BinaryFieldProfile;
    for owners in [1, 158, 4096] {
        let manifest = binary_word_manifest_fixture(BinaryFieldProfile::P64, owners);
        let typed = BinaryTypedPrefix::new(&manifest, true, 0, 64).unwrap();
        assert_eq!(
            typed.rows,
            [
                manifest.b_rows(),
                manifest.d_label(true, 0).unwrap().rank as usize
            ]
        );
        for (role, &rows) in typed.rows.iter().enumerate() {
            assert!(typed
                .pieces
                .iter()
                .filter(|p| typed.segments[p.segment].role == role)
                .all(|p| p.row < rows));
            assert!(typed
                .pieces
                .iter()
                .any(|p| typed.segments[p.segment].role == role && p.row + 1 == rows));
        }
        assert!(typed
            .segments
            .iter()
            .all(|s| s.lanes.digits() != 8 || s.role == 0));
    }
}
