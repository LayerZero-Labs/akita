use super::*;
use crate::root_repacking::{RootCoefficientRepacking, RootRepackingSegment};
use akita_algebra::offset_eq::eq_eval_at_index;
use akita_params::binary_setup::BinaryFieldProfile;
use akita_params::RelationRowGeometry;
use jolt_field::{Ext2, ExtField, One, Prime64Offset59 as Base, Ring, Zero};
use rand::{rngs::StdRng, Rng, SeedableRng};

type E = Ext2<Base>;

fn extension(rng: &mut StdRng) -> E {
    E::from_base_fn(|_| Base::from_u64(rng.gen_range(1..u64::MAX)))
}

fn manifest(
    rows: usize,
    columns: usize,
    owners: usize,
    group: u64,
) -> BinaryCommittedGroupManifest {
    let index =
        crate::binary::BinaryIndexMap::new(128, 4 * rows * columns, 4 * rows * columns, columns)
            .unwrap();
    let profiles = BinaryCommittedGroupManifest::profiles_for_field(
        &index,
        owners,
        group,
        BinaryFieldProfile::P64,
    )
    .unwrap();
    let descriptor = crate::binary::BinarySourceDescriptor::new(
        index,
        (0..owners).map(|i| [i as u8 + 1; 32]).collect(),
        profiles,
    )
    .unwrap();
    BinaryCommittedGroupManifest::with_field_profile(
        descriptor,
        group,
        &crate::binary::binary_ordinary_profile_fixture(BinaryFieldProfile::P64),
        BinaryFieldProfile::P64,
    )
    .unwrap()
}

fn expand(blocks: &TypedPrefixWeights<E>) -> Vec<E> {
    let mut dense = vec![E::zero(); blocks.domain_len()];
    for (live, &block) in blocks.blocks().iter().enumerate() {
        for (scale, row) in blocks.block_terms(live).unwrap() {
            for (slot, &value) in row.iter().enumerate() {
                dense[block * 64 + slot] += scale * value;
            }
        }
    }
    dense
}

// Independent dense address equality for every stored setup column copy.
fn dense_weight(
    typed: &BinaryTypedPrefix,
    translated: &BinaryPrefixTranslation<'_>,
    a: &BinaryAProduct<E>,
    root_point: &[E],
    alpha: E,
    batches: [&[E]; 2],
) -> Vec<E> {
    let mut weight = vec![E::zero(); typed.domain_len];
    for (index, coefficient) in a.factor_terms().unwrap() {
        for (row, &value) in index.iter().take(typed.a.rows).enumerate() {
            for (c, &coefficient) in coefficient.iter().enumerate() {
                weight[row * typed.a.row_stride + c] += value * coefficient;
            }
        }
    }
    let kernel = ResidueKernelPoint::new(alpha, SLOTS).unwrap();
    for cell in (0..typed.natural_len).step_by(SLOTS) {
        let Some((segment, row, q, j)) = typed.locate(cell).unwrap() else {
            continue;
        };
        let segment_index = typed.segments.iter().position(|s| *s == segment).unwrap();
        let mut eq = [E::zero(); SLOTS];
        for (slot, value) in eq.iter_mut().enumerate() {
            if let Some(address) = segment.address(segment.lanes.start(), q, j, slot) {
                let address = translated.starts[segment_index] + address - segment.lanes.start();
                *value = eq_eval_at_index(root_point, address);
            }
        }
        kernel
            .field_kernel_into(&eq, &mut weight[cell..cell + SLOTS])
            .unwrap();
        for value in &mut weight[cell..cell + SLOTS] {
            *value *= batches[segment.role][row];
        }
    }
    weight
}

#[test]
fn translated_two_and_four_setup_weights_match_dense_e64() {
    let mut rng = StdRng::seed_from_u64(0x4a_0004);
    let shapes = [
        (1, 1, 1, true, 0),
        (2, 2, 2, true, 1),
        (4, 1, 3, false, 3),
        (1, 4, 1, true, 1),
    ];
    for groups in [2, 4] {
        let manifests = shapes[..groups]
            .iter()
            .enumerate()
            .map(|(i, &(r, c, o, _, _))| manifest(r, c, o, i as u64))
            .collect::<Vec<_>>();
        let plans = manifests
            .iter()
            .zip(&shapes)
            .map(|(m, &(_, _, _, w, claims))| {
                BinaryRootWitnessPlan::for_profile(
                    m.descriptor(),
                    3,
                    64,
                    BinaryFieldProfile::P64,
                    w,
                    claims * BinaryFieldProfile::P64.extension_degree(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let locals = manifests
            .iter()
            .zip(&plans)
            .map(|(m, p)| BinaryTypedPrefix::from_plan(m, p).unwrap())
            .collect::<Vec<_>>();
        assert_ne!(locals[0].rows, locals[1].rows);
        // A short opaque segment deliberately makes the first block cross a
        // binary carry chain. All quotients remain in the final bank.
        let mut segments = vec![RootRepackingSegment {
            geometry: RelationRowGeometry::native(1).unwrap(),
            elements: 511,
            digits: 1,
        }];
        let mut maps = vec![Vec::new(); groups];
        for (map, plan) in maps.iter_mut().zip(&plans) {
            for segment in &plan.repacking().segments()[..plan.repacking().segments().len() - 1] {
                map.push(segments.len());
                segments.push(*segment);
            }
        }
        for (map, plan) in maps.iter_mut().zip(&plans) {
            map.push(segments.len());
            segments.push(*plan.repacking().segments().last().unwrap());
        }
        let root = RootCoefficientRepacking::new(segments, 64).unwrap();
        let root_point = (0..root.domain_len().trailing_zeros())
            .map(|_| extension(&mut rng))
            .collect::<Vec<_>>();
        assert!(root_point
            .iter()
            .all(|p| p.base_coefficient(1) != Base::zero()));
        use crate::binary::composite_prefix::tests::slot;
        use crate::binary::{
            composite_prefix_root_identity, CompositePrefixBlock, CompositePrefixLayout,
        };
        let blocks = locals
            .iter()
            .zip(&manifests)
            .enumerate()
            .map(|(i, (typed, m))| {
                CompositePrefixBlock::binary(
                    i,
                    &[i as u8 + 20; 32],
                    m,
                    typed,
                    &slot(typed.domain_len),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let mut identity_maps = maps.clone();
        identity_maps[0].insert(0, 0);
        let owners = shapes[..groups]
            .iter()
            .map(|&(_, _, _, _, p)| (0..p).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let root_identity = composite_prefix_root_identity(
            &root,
            &identity_maps.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            &owners.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        )
        .unwrap();
        let end = blocks.iter().fold(0usize, |end, b| {
            end.div_ceil(b.domain_len) * b.domain_len + b.domain_len
        });
        let composite =
            CompositePrefixLayout::new(&blocks, &root_identity, &slot(end.next_power_of_two()))
                .unwrap();
        let mut other_owners = owners.clone();
        other_owners[1][0] = 1;
        let subset_identity = composite_prefix_root_identity(
            &root,
            &identity_maps.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            &other_owners.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        )
        .unwrap();
        assert_ne!(
            composite.identity(),
            CompositePrefixLayout::new(&blocks, &subset_identity, composite.slot())
                .unwrap()
                .identity()
        );
        let other_ring = RootCoefficientRepacking::new(root.segments().to_vec(), 128)
            .unwrap()
            .with_domain_len(root.domain_len())
            .unwrap();
        let ring_identity = composite_prefix_root_identity(
            &other_ring,
            &identity_maps.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            &owners.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        )
        .unwrap();
        assert_ne!(
            composite.identity(),
            CompositePrefixLayout::new(&blocks, &ring_identity, composite.slot())
                .unwrap()
                .identity()
        );

        let composite_point = (0..composite.domain_len().trailing_zeros())
            .map(|_| extension(&mut rng))
            .collect::<Vec<_>>();
        let mut local_weights = Vec::new();
        let mut local_setups = Vec::new();
        let mut local_evaluations = Vec::new();
        let mut product_claim = E::zero();
        for (group, ((typed, plan), manifest)) in
            locals.iter().zip(&plans).zip(&manifests).enumerate()
        {
            let translated = typed
                .translated(manifest, plan, &root, &maps[group])
                .unwrap();
            assert_ne!(translated.response_start() % 1024, 0);
            let rho = (0..manifest.descriptor().polynomials().len() * 3)
                .map(|_| std::array::from_fn(|_| extension(&mut rng)))
                .collect::<Vec<_>>();
            let a = BinaryAProduct::translated(
                plan,
                &rho,
                &root_point,
                typed.domain_len,
                translated.response_start(),
            )
            .unwrap();
            let batches = typed
                .rows
                .map(|n| (0..n).map(|_| extension(&mut rng)).collect::<Vec<_>>());
            let refs = [&batches[0][..], &batches[1][..]];
            let alpha = extension(&mut rng);
            let dense = dense_weight(typed, &translated, &a, &root_point, alpha, refs);
            let blocks = translated
                .prover_weight_blocks(&a, &root_point, alpha, refs, typed.domain_len)
                .unwrap();
            assert_eq!(expand(&blocks), dense);
            let point = composite_point[..typed.domain_len.trailing_zeros() as usize].to_vec();
            let dense_at_point = dense
                .iter()
                .zip(EqPolynomial::evals(&point).unwrap())
                .fold(E::zero(), |sum, (&v, e)| sum + v * e);
            assert_eq!(
                a.evaluate(&point).unwrap()
                    + translated
                        .digit_weight(&point, &root_point, alpha, refs)
                        .unwrap(),
                dense_at_point
            );
            let setup = typed
                .setup_coefficients::<Base>(&[group as u8 + 20; 32], manifest, typed.domain_len)
                .unwrap();
            let dense_dot = setup
                .iter()
                .zip(&dense)
                .fold(E::zero(), |sum, (&s, &f)| sum + E::lift_base(s) * f);
            let block_dot =
                blocks
                    .blocks()
                    .iter()
                    .enumerate()
                    .fold(E::zero(), |sum, (live, &block)| {
                        sum + blocks.block_terms(live).unwrap().iter().fold(
                            E::zero(),
                            |sum, (scale, row)| {
                                sum + row.iter().enumerate().fold(E::zero(), |sum, (slot, &v)| {
                                    sum + *scale * v * E::lift_base(setup[block * 64 + slot])
                                })
                            },
                        )
                    });
            assert_eq!(dense_dot, block_dot);
            assert_ne!(dense_dot.base_coefficient(1), Base::zero());
            product_claim += dense_dot;
            local_evaluations.push(
                a.evaluate(&point).unwrap()
                    + translated
                        .digit_weight(&point, &root_point, alpha, refs)
                        .unwrap(),
            );
            local_weights.push(dense);
            local_setups.push(setup);
        }
        let weights = composite
            .assemble(
                &local_weights.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                composite.domain_len(),
            )
            .unwrap();
        let setup = composite
            .assemble(
                &local_setups.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                composite.domain_len(),
            )
            .unwrap();
        let dense_product = setup
            .iter()
            .zip(&weights)
            .fold(E::zero(), |sum, (&s, &w)| sum + E::lift_base(s) * w);
        assert_eq!(product_claim, dense_product);
        let dense_extension = weights
            .iter()
            .zip(EqPolynomial::evals(&composite_point).unwrap())
            .fold(E::zero(), |sum, (&w, eq)| sum + w * eq);
        assert_eq!(
            composite
                .evaluate(&composite_point, &local_evaluations)
                .unwrap(),
            dense_extension
        );
        let mut covered = vec![false; composite.domain_len()];
        for placement in composite.placements() {
            covered[placement.offset..placement.offset + placement.block.domain_len].fill(true);
        }
        assert!(covered
            .iter()
            .zip(&weights)
            .zip(&setup)
            .all(|((&live, &w), &s)| live || (w == E::zero() && s == Base::zero())));
        // Reassigning the first two local tables changes their roles in F and
        // the composite identity, even with the same root point and row batches.
        let mut swapped = blocks.clone();
        swapped.swap(0, 1);
        swapped[0].index = 0;
        swapped[1].index = 1;
        let swapped_end = swapped.iter().fold(0usize, |end, b| {
            end.div_ceil(b.domain_len) * b.domain_len + b.domain_len
        });
        let other = CompositePrefixLayout::new(
            &swapped,
            &root_identity,
            &slot(swapped_end.next_power_of_two()),
        )
        .unwrap();
        assert_ne!(composite.identity(), other.identity());
        let mut swapped_weights = local_weights.clone();
        swapped_weights.swap(0, 1);
        let other_weights = other
            .assemble(
                &swapped_weights
                    .iter()
                    .map(Vec::as_slice)
                    .collect::<Vec<_>>(),
                other.domain_len(),
            )
            .unwrap();
        assert_ne!(weights, other_weights);
    }
}

#[test]
fn offset_equality_carries_match_dense_nonbase_points() {
    let mut rng = StdRng::seed_from_u64(917);
    for bits in [1, 4, 8, 12] {
        let a = (0..bits).map(|_| extension(&mut rng)).collect::<Vec<_>>();
        let b = (0..bits + 1)
            .map(|_| extension(&mut rng))
            .collect::<Vec<_>>();
        for j in 0..=bits {
            let offset = (1 << j) - 1;
            for count in [1, 3, (1 << bits) - 1] {
                let expected = (0..count).fold(E::zero(), |s, i| {
                    s + eq_eval_at_index(&a, i) * eq_eval_at_index(&b, offset + i)
                });
                assert_eq!(
                    paired_offset_eq_sum(&a, 0, &b, offset, count).unwrap(),
                    expected
                );
            }
        }
    }
}

#[test]
fn zero_translation_preserves_all_local_weights_and_rejects_bad_maps() {
    let m = manifest(2, 2, 2, 0);
    let plan =
        BinaryRootWitnessPlan::for_profile(m.descriptor(), 3, 64, BinaryFieldProfile::P64, true, 2)
            .unwrap();
    let typed = BinaryTypedPrefix::from_plan(&m, &plan).unwrap();
    let indices = (0..plan.repacking().segments().len()).collect::<Vec<_>>();
    let translated = typed
        .translated(&m, &plan, plan.repacking(), &indices)
        .unwrap();
    let mut rng = StdRng::seed_from_u64(0x1011);
    let r = (0..plan.repacking().domain_len().trailing_zeros())
        .map(|_| extension(&mut rng))
        .collect::<Vec<_>>();
    let rho = (0..6)
        .map(|_| std::array::from_fn(|_| extension(&mut rng)))
        .collect::<Vec<_>>();
    let a = BinaryAProduct::new(&plan, &rho, &r, typed.domain_len).unwrap();
    let batches = typed
        .rows
        .map(|n| (0..n).map(|_| extension(&mut rng)).collect::<Vec<_>>());
    let refs = [&batches[0][..], &batches[1][..]];
    let alpha = extension(&mut rng);
    assert!(translated
        .digit_weight(&[E::one(); 33], &r, alpha, refs)
        .is_err());
    if let Some(oversized) = checked::pow2(33) {
        assert!(translated
            .prover_weight_blocks(&a, &r, alpha, refs, oversized)
            .is_err());
    }
    assert_eq!(
        expand(
            &translated
                .prover_weight_blocks(&a, &r, alpha, refs, typed.domain_len)
                .unwrap()
        ),
        typed
            .prover_weights(&a, &r, alpha, refs, typed.domain_len)
            .unwrap()
    );
    let z = (0..typed.domain_len.trailing_zeros())
        .map(|_| extension(&mut rng))
        .collect::<Vec<_>>();
    assert_eq!(
        translated.digit_weight(&z, &r, alpha, refs).unwrap(),
        typed.digit_weight(&z, &r, alpha, refs).unwrap()
    );
    assert!(typed.translated(&m, &plan, plan.repacking(), &[]).is_err());
    let mut bad = indices.clone();
    bad[1] = bad[0];
    assert!(typed.translated(&m, &plan, plan.repacking(), &bad).is_err());
    assert!(translated
        .digit_weight(&z, &r[..r.len() - 1], alpha, refs)
        .is_err());
    assert!(translated
        .digit_weight(&z[..z.len() - 1], &r, alpha, refs)
        .is_err());
}
