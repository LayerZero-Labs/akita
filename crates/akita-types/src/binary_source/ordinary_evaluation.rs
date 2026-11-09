//! Source reading weights reused at each ordinary point.

use super::{ContiguousLanes, OrdinaryNativeRootPlan, SourceEvaluationWeights};
use crate::native_root_relation::{NativeRootWeights, RootWeightFamily};
use crate::root_repacking::RootCoefficientRepacking;
use akita_algebra::offset_eq::OffsetEqWindow;
use akita_error::{checked, AkitaError};
use jolt_field::Field;

impl OrdinaryNativeRootPlan {
    /// Compile the source evaluation equation at one point for every signed slot.
    pub fn evaluation_weights<E: Field>(
        &self,
        repacking: &RootCoefficientRepacking,
        point: &[E],
        evaluations: &[E],
        batches: &[E],
    ) -> Result<NativeRootWeights<E>, AkitaError> {
        self.validate_repacking(repacking)?;
        let owners = self.profile.group.num_polynomials();
        if point.len() != self.profile.group.num_vars()
            || evaluations.len() != owners
            || batches.len() != owners
        {
            return Err(AkitaError::InvalidProof);
        }
        let table_len = repacking.domain_len();
        let s_lanes = ContiguousLanes::new(repacking, self.first_segment)?;
        let a = self.profile.inner.matrix;
        let positions = self.profile.blocks.positions_per_block;
        let source_digits = self.profile.inner.digits.num_digits;
        let mut families = Vec::new();
        let mut terms = Vec::new();
        let source_len = checked::pow2(point.len()).ok_or(AkitaError::InvalidProof)?;
        let owner_coefficients = checked::product([
            self.profile.blocks.live_blocks,
            positions,
            a.ring_dimension(),
        ])
        .ok_or(AkitaError::InvalidProof)?;
        for (owner, &batch) in batches.iter().enumerate() {
            let start =
                checked::product([owner, owner_coefficients]).ok_or(AkitaError::InvalidProof)?;
            if source_digits == 1 {
                families.push(RootWeightFamily::SourceEvaluation(
                    SourceEvaluationWeights::new(
                        table_len,
                        s_lanes.contiguous(start, source_len)?,
                        point,
                        batch,
                    )?,
                ));
            } else {
                // Multi-digit sources interleave digits, so the slot's
                // evaluation is not one address interval; keep its terms.
                let equality = OffsetEqWindow::new(point)?;
                for coefficient in 0..source_len {
                    let element = checked::sum([start, coefficient])
                        .map(|flat| flat / a.ring_dimension())
                        .ok_or(AkitaError::InvalidProof)?;
                    terms.extend(repacking.coefficient_weights(
                        self.first_segment,
                        element,
                        coefficient % a.ring_dimension(),
                        batch * equality.eval(coefficient),
                    )?);
                }
            }
        }
        let claim = batches
            .iter()
            .zip(evaluations)
            .fold(E::zero(), |sum, (&batch, &value)| sum + batch * value);
        NativeRootWeights::structured(table_len, terms, families, claim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binary_source::{binary_ordinary_profile_fixture, BinaryRootPublic};
    use akita_params::binary_setup::BinaryFieldProfile;
    use jolt_field::{Ext2, ExtField, One, Prime64Offset59 as F, Ring, Zero};

    type E = Ext2<F>;

    #[test]
    fn each_ordinary_reading_constrains_the_same_source() {
        let profile = binary_ordinary_profile_fixture(BinaryFieldProfile::P64, None);
        let plan = OrdinaryNativeRootPlan::new(profile, 0).unwrap();
        let repacking = RootCoefficientRepacking::new(plan.segments().to_vec(), 64).unwrap();
        let mut source = vec![E::zero(); repacking.domain_len()];
        let segment = repacking.segment_range(0).unwrap();
        for (i, value) in source[segment].iter_mut().enumerate() {
            *value = E::from_u64((i & 1) as u64);
        }
        let points: [Vec<E>; 2] = std::array::from_fn(|copy| {
            (0..profile.group.num_vars())
                .map(|axis| {
                    E::from_base_fn(|limb| {
                        F::from_u64((11 + copy * 17 + axis * 3 + limb * 7) as u64)
                    })
                })
                .collect()
        });
        let batches = [E::one(), E::from_u64(3)];
        let weights = |values: [E; 2]| {
            NativeRootWeights::merge(
                repacking.domain_len(),
                (0..2).map(|i| {
                    plan.evaluation_weights(&repacking, &points[i], &[values[i]], &[batches[i]])
                        .unwrap()
                }),
                E::zero(),
            )
            .unwrap()
        };
        let values = [points[0][0], points[1][0]];
        weights(values).check_table(&source).unwrap();
        for i in 0..2 {
            let mut wrong = values;
            wrong[i] += E::one();
            assert!(weights(wrong).check_table(&source).is_err());
        }
        assert!(weights([values[1], values[0]])
            .check_table(&source)
            .is_err());
        assert!(plan
            .evaluation_weights(&repacking, &points[0], &[], &[E::one()])
            .is_err());
        assert!(plan
            .evaluation_weights(&repacking, &points[1][1..], &[E::zero()], &[E::one()])
            .is_err());
    }

    #[test]
    fn second_point_changes_only_relation_coordinates() {
        let manifest =
            crate::binary_source::binary_word_manifest_fixture(BinaryFieldProfile::P64, 1);
        let prime = crate::binary_source::PrimePlaneManifest::with_field_profile(
            manifest.descriptor(),
            Some(crate::binary_source::PrimeGroupClaim {
                point: vec![[2, 0].into(); 11],
                evaluations: vec![crate::binary_source::PrimePlaneClaim {
                    polynomial: 0,
                    value: [0, 0].into(),
                }],
            }),
            BinaryFieldProfile::P64,
        )
        .unwrap();
        let setup = crate::AkitaSetupDescriptor {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            num_field_elements: crate::MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS,
            setup_seed: [41; 32].into(),
        };
        let profile = binary_ordinary_profile_fixture(BinaryFieldProfile::P64, None);
        let single =
            BinaryRootPublic::new(&manifest, None, &prime, &setup, 1 << 17, 64, Some(profile))
                .unwrap();
        let two = single.clone().with_ordinary_point_count(2).unwrap();
        assert_eq!(single.plan().repacking(), two.plan().repacking());
        assert_eq!(two.batch_count(), single.batch_count() + 1);
        assert_eq!(
            two.clone()
                .with_ordinary_point_count(1)
                .unwrap()
                .batch_count(),
            single.batch_count()
        );
        assert!(single.clone().with_ordinary_point_count(0).is_err());
        assert!(single.with_ordinary_point_count(3).is_err());
    }
}
