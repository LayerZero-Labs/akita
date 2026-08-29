//! Checked geometry for a flat Stage-2 relation evaluation point.
//!
//! This is the single authority for splitting coefficient coordinates from
//! lane-and-column coordinates and preparing either lifted role powers or the
//! complete reduced terminal functional.

#[cfg(test)]
use akita_algebra::poly::multilinear_eval;
use akita_algebra::ring::{evaluate_power_sequence_mle, scalar_powers};
use akita_error::AkitaError;
#[cfg(test)]
use akita_types::{CommitmentRingDims, RingRole};
use akita_types::{
    PreparedCoefficientFunctional, PreparedRelationAddress, RelationAddressGeometry,
    RingRelationMode,
};
use jolt_field::Field;
use std::sync::Arc;

pub(super) struct PreparedRolePoint<E: Field> {
    pub(super) ring_dim: usize,
    pub(super) powers: Arc<[E]>,
    pub(super) lane_powers: Arc<[E]>,
}

enum PreparedCoefficientPoint<E: Field> {
    Lifted {
        common_alpha_evaluation: E,
        inner: Arc<PreparedRolePoint<E>>,
        outer: Arc<PreparedRolePoint<E>>,
        opening: Arc<PreparedRolePoint<E>>,
        additional: Vec<Arc<PreparedRolePoint<E>>>,
    },
    Reduced {
        functional: PreparedCoefficientFunctional<E>,
    },
}

/// Checked factorization of one flat Stage-2 relation point.
///
/// `coeff_count` is the low-address block shared by the current relation
/// roles. The remaining point addresses relation lanes followed by semantic
/// witness columns. Role-native setup columns split one A-role witness column
/// into `d_a / d_role` subcolumns.
pub(super) struct PreparedRelationPoint<E: Field> {
    relation_address_geometry: RelationAddressGeometry,
    alpha: E,
    relation_address: PreparedRelationAddress<E>,
    #[cfg(test)]
    role_dims: CommitmentRingDims,
    coefficient_point: PreparedCoefficientPoint<E>,
}

impl<E: Field> PreparedRelationPoint<E> {
    pub(super) fn new(
        point: &[E],
        alpha: E,
        geometry: RelationAddressGeometry,
        additional_ring_dims: &[usize],
        mode: RingRelationMode,
    ) -> Result<Self, AkitaError> {
        let role_dims = geometry.role_dims();
        geometry.validate_relation_point_len(point.len())?;
        let coeff_count = geometry.relation_coefficient_block_len();

        let coeff_bits = geometry.relation_coefficient_variable_count();
        let coeff_point = point.get(..coeff_bits).ok_or(AkitaError::InvalidProof)?;
        let lane_and_column_point = point.get(coeff_bits..).ok_or(AkitaError::InvalidProof)?;
        let relation_address = PreparedRelationAddress::new(lane_and_column_point)?;
        let coefficient_point = match mode {
            RingRelationMode::QuotientLift => {
                let prepare_role =
                    |ring_dim: usize| -> Result<Arc<PreparedRolePoint<E>>, AkitaError> {
                        if !ring_dim.is_power_of_two() || !ring_dim.is_multiple_of(coeff_count) {
                            return Err(AkitaError::InvalidSetup(
                                "relation role dimension is not aligned to the common coefficient block"
                                    .into(),
                            ));
                        }
                        let lane_count = ring_dim
                            .checked_div(coeff_count)
                            .filter(|&count| count != 0)
                            .ok_or_else(|| {
                                AkitaError::InvalidSetup("invalid relation role lane count".into())
                            })?;
                        let powers: Arc<[E]> = scalar_powers(alpha, ring_dim).into();
                        let lane_powers = (0..lane_count)
                            .map(|lane| {
                                powers
                                    .get(lane * coeff_count)
                                    .copied()
                                    .ok_or(AkitaError::InvalidProof)
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        Ok(Arc::new(PreparedRolePoint {
                            ring_dim,
                            powers,
                            lane_powers: lane_powers.into(),
                        }))
                    };

                let inner = prepare_role(role_dims.d_a())?;
                let outer = if role_dims.d_b() == role_dims.d_a() {
                    inner.clone()
                } else {
                    prepare_role(role_dims.d_b())?
                };
                let opening = if role_dims.d_d() == role_dims.d_a() {
                    inner.clone()
                } else if role_dims.d_d() == role_dims.d_b() {
                    outer.clone()
                } else {
                    prepare_role(role_dims.d_d())?
                };
                let mut additional = Vec::new();
                for &ring_dim in additional_ring_dims {
                    if ring_dim == 0
                        || !ring_dim.is_power_of_two()
                        || !ring_dim.is_multiple_of(coeff_count)
                    {
                        return Err(AkitaError::InvalidSetup(format!(
                            "relation quotient ring dimension {ring_dim} does not fit common block {coeff_count}",
                        )));
                    }
                    if [role_dims.d_a(), role_dims.d_b(), role_dims.d_d()].contains(&ring_dim)
                        || additional
                            .iter()
                            .any(|prepared: &Arc<PreparedRolePoint<E>>| {
                                prepared.ring_dim == ring_dim
                            })
                    {
                        continue;
                    }
                    additional.push(prepare_role(ring_dim)?);
                }
                PreparedCoefficientPoint::Lifted {
                    common_alpha_evaluation: evaluate_power_sequence_mle(alpha, coeff_point),
                    inner,
                    outer,
                    opening,
                    additional,
                }
            }
            RingRelationMode::ReducedEvaluation => {
                if !additional_ring_dims.is_empty() {
                    return Err(AkitaError::InvalidSetup(
                        "reduced relation point must not prepare quotient dimensions".into(),
                    ));
                }
                PreparedCoefficientPoint::Reduced {
                    functional: PreparedCoefficientFunctional::reduced_evaluation(
                        alpha,
                        coeff_point,
                        geometry,
                    )?,
                }
            }
        };

        Ok(Self {
            relation_address_geometry: geometry,
            alpha,
            relation_address,
            #[cfg(test)]
            role_dims,
            coefficient_point,
        })
    }

    pub(super) const fn relation_address_geometry(&self) -> RelationAddressGeometry {
        self.relation_address_geometry
    }

    pub(super) fn common_alpha_evaluation(&self) -> Result<E, AkitaError> {
        match &self.coefficient_point {
            PreparedCoefficientPoint::Lifted {
                common_alpha_evaluation,
                ..
            } => Ok(*common_alpha_evaluation),
            PreparedCoefficientPoint::Reduced { .. } => Err(AkitaError::InvalidProof),
        }
    }

    pub(super) fn coefficient_functional(&self) -> PreparedCoefficientFunctional<E> {
        match &self.coefficient_point {
            PreparedCoefficientPoint::Lifted { .. } => {
                PreparedCoefficientFunctional::lifted_power(self.alpha)
            }
            PreparedCoefficientPoint::Reduced { functional } => functional.clone(),
        }
    }

    pub(super) fn alpha(&self) -> E {
        self.alpha
    }

    pub(super) fn address_point(&self) -> &[E] {
        self.relation_address.point()
    }

    pub(super) fn relation_address(&self) -> &PreparedRelationAddress<E> {
        &self.relation_address
    }

    pub(super) fn for_dimension(
        &self,
        ring_dim: usize,
    ) -> Result<&PreparedRolePoint<E>, AkitaError> {
        let PreparedCoefficientPoint::Lifted {
            inner,
            outer,
            opening,
            additional,
            ..
        } = &self.coefficient_point
        else {
            return Err(AkitaError::InvalidProof);
        };
        [inner.as_ref(), outer.as_ref(), opening.as_ref()]
            .into_iter()
            .chain(additional.iter().map(Arc::as_ref))
            .find(|role| role.ring_dim == ring_dim)
            .ok_or(AkitaError::InvalidProof)
    }

    /// Evaluate the high-address factor for one role-native setup column.
    #[cfg(test)]
    pub(super) fn role_column_weight(
        &self,
        witness_column: usize,
        role: RingRole,
        role_subcolumn: usize,
    ) -> Result<E, AkitaError> {
        let prepared = self.role(role)?;
        let subcolumn_count = self
            .role_dims
            .d_a()
            .checked_div(prepared.ring_dim)
            .filter(|&count| count != 0)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("invalid relation role subcolumn count".into())
            })?;
        if role_subcolumn >= subcolumn_count {
            return Err(AkitaError::InvalidProof);
        }

        let physical_start = witness_column
            .checked_mul(self.role_dims.d_a())
            .and_then(|start| {
                role_subcolumn
                    .checked_mul(prepared.ring_dim)
                    .and_then(|offset| start.checked_add(offset))
            })
            .ok_or_else(|| AkitaError::InvalidSetup("relation role address overflow".into()))?;
        let physical_end = physical_start
            .checked_add(prepared.ring_dim)
            .ok_or_else(|| AkitaError::InvalidSetup("relation role address overflow".into()))?;
        if physical_end
            > self
                .relation_address_geometry
                .digit_witness_domain()
                .live_len()
        {
            return Err(AkitaError::InvalidInput(
                "flat relation witness address out of range".into(),
            ));
        }
        let coeff_count = self
            .relation_address_geometry
            .relation_coefficient_block_len();
        if !physical_start.is_multiple_of(coeff_count) {
            return Err(AkitaError::InvalidProof);
        }
        let lane_start = physical_start / coeff_count;

        prepared.lane_powers.iter().copied().enumerate().try_fold(
            E::zero(),
            |evaluation, (lane, alpha_power)| {
                let address = lane_start.checked_add(lane).ok_or_else(|| {
                    AkitaError::InvalidSetup("relation lane address overflow".into())
                })?;
                Ok(
                    evaluation
                        + self.relation_address.equality_window().eval(address) * alpha_power,
                )
            },
        )
    }

    #[cfg(test)]
    fn role(&self, role: RingRole) -> Result<&PreparedRolePoint<E>, AkitaError> {
        let PreparedCoefficientPoint::Lifted {
            inner,
            outer,
            opening,
            ..
        } = &self.coefficient_point
        else {
            return Err(AkitaError::InvalidProof);
        };
        Ok(match role {
            RingRole::Inner => inner,
            RingRole::Outer => outer,
            RingRole::Opening => opening,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Prime128OffsetA7F7;
    use jolt_field::{One, Ring, Zero};

    type F = Prime128OffsetA7F7;

    fn point_for(field_len: usize) -> Vec<F> {
        (0..field_len.trailing_zeros() as usize)
            .map(|index| F::from_u64(17 + index as u64))
            .collect()
    }

    fn assert_role_columns_match_dense(
        role_dims: CommitmentRingDims,
        outgoing_ring_dim: usize,
        alpha: F,
    ) {
        let flat_live_len = 1024;
        let geometry =
            RelationAddressGeometry::new(role_dims, outgoing_ring_dim, flat_live_len).unwrap();
        let field_len = geometry.committed_witness_coeff_len();
        let point = point_for(field_len);
        let prepared = PreparedRelationPoint::new(
            &point,
            alpha,
            geometry,
            &[],
            RingRelationMode::QuotientLift,
        )
        .unwrap();

        assert_eq!(
            prepared
                .relation_address_geometry()
                .relation_coefficient_block_len(),
            role_dims.common_relation_coeff_count()
        );
        for role in [RingRole::Inner, RingRole::Outer, RingRole::Opening] {
            let role_dim = role_dims.dim_for(role);
            let subcolumn_count = role_dims.d_a() / role_dim;
            for witness_column in 0..2 {
                for role_subcolumn in 0..subcolumn_count {
                    let physical_start =
                        witness_column * role_dims.d_a() + role_subcolumn * role_dim;
                    let alpha_powers = scalar_powers(alpha, role_dim);
                    let mut dense = vec![F::zero(); field_len];
                    for (offset, alpha_power) in alpha_powers.into_iter().enumerate() {
                        dense[physical_start + offset] = alpha_power;
                    }
                    let expected = multilinear_eval(&dense, &point).unwrap();
                    let got = prepared.common_alpha_evaluation().unwrap()
                        * prepared
                            .role_column_weight(witness_column, role, role_subcolumn)
                            .unwrap();
                    assert_eq!(
                        got, expected,
                        "role={role:?} witness_column={witness_column} subcolumn={role_subcolumn}"
                    );
                }
            }
        }
    }

    #[test]
    fn prepared_relation_point_matches_dense_role_columns() {
        let geometries = [
            (CommitmentRingDims::uniform(128), 128),
            (CommitmentRingDims::uniform(128), 64),
            (
                CommitmentRingDims {
                    inner: 128,
                    outer: 64,
                    opening: 64,
                },
                16,
            ),
            (
                CommitmentRingDims {
                    inner: 128,
                    outer: 64,
                    opening: 64,
                },
                32,
            ),
            (
                CommitmentRingDims {
                    inner: 128,
                    outer: 64,
                    opening: 64,
                },
                64,
            ),
            (
                CommitmentRingDims {
                    inner: 128,
                    outer: 64,
                    opening: 64,
                },
                32,
            ),
        ];
        for (role_dims, outgoing_ring_dim) in geometries {
            for alpha in [F::zero(), F::one(), F::from_u64(7)] {
                assert_role_columns_match_dense(role_dims, outgoing_ring_dim, alpha);
            }
        }
    }

    #[test]
    fn prepared_relation_point_rejects_malformed_geometry() {
        let role_dims = CommitmentRingDims::uniform(128);
        assert!(matches!(
            RelationAddressGeometry::new(role_dims, 0, 256),
            Err(AkitaError::InvalidSetup(_))
        ));
        let geometry = RelationAddressGeometry::new(role_dims, 128, 256).unwrap();
        let point = point_for(geometry.committed_witness_coeff_len());
        assert!(matches!(
            PreparedRelationPoint::new(
                &point[..point.len() - 1],
                F::one(),
                geometry,
                &[],
                RingRelationMode::QuotientLift,
            ),
            Err(AkitaError::InvalidSize { .. })
        ));
        let invalid_roles = CommitmentRingDims {
            inner: 64,
            outer: 128,
            opening: 64,
        };
        assert!(matches!(
            RelationAddressGeometry::new(invalid_roles, 128, 256),
            Err(AkitaError::InvalidSetup(_))
        ));
    }

    #[test]
    fn reduced_relation_point_owns_only_the_terminal_coefficient_functional() {
        let role_dims = CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 64,
        };
        let geometry = RelationAddressGeometry::new(role_dims, 32, 320).unwrap();
        let point = point_for(geometry.committed_witness_coeff_len());
        let prepared = PreparedRelationPoint::new(
            &point,
            F::from_u64(7),
            geometry,
            &[],
            RingRelationMode::ReducedEvaluation,
        )
        .unwrap();
        assert!(matches!(
            prepared.coefficient_functional(),
            PreparedCoefficientFunctional::ReducedEvaluation { .. }
        ));
        assert!(prepared.common_alpha_evaluation().is_err());
        assert!(prepared.for_dimension(role_dims.d_a()).is_err());
        assert!(PreparedRelationPoint::new(
            &point,
            F::from_u64(7),
            geometry,
            &[64],
            RingRelationMode::ReducedEvaluation,
        )
        .is_err());
    }

    #[test]
    fn prepared_relation_point_rejects_out_of_range_witness_columns() {
        let role_dims = CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 64,
        };
        let live_witness_coeff_len = 320;
        let outgoing_ring_dim = 32;
        let geometry =
            RelationAddressGeometry::new(role_dims, outgoing_ring_dim, live_witness_coeff_len)
                .unwrap();
        let point = point_for(geometry.committed_witness_coeff_len());
        let prepared = PreparedRelationPoint::new(
            &point,
            F::from_u64(7),
            geometry,
            &[],
            RingRelationMode::QuotientLift,
        )
        .unwrap();
        assert!(matches!(
            prepared.role_column_weight(9, RingRole::Inner, 0),
            Err(AkitaError::InvalidInput(_))
        ));
        assert!(matches!(
            prepared.role_column_weight(0, RingRole::Outer, 2),
            Err(AkitaError::InvalidProof)
        ));
    }

    #[test]
    fn prepares_additional_group_local_quotient_dimension() {
        let role_dims = CommitmentRingDims {
            inner: 256,
            outer: 128,
            opening: 128,
        };
        let geometry = RelationAddressGeometry::new_for_groups(
            role_dims,
            &[CommitmentRingDims::uniform(64)],
            32,
            320,
        )
        .unwrap();
        let point = point_for(geometry.committed_witness_coeff_len());
        let prepared = PreparedRelationPoint::new(
            &point,
            F::from_u64(7),
            geometry,
            &[64, 64],
            RingRelationMode::QuotientLift,
        )
        .expect("additional group-local dimension");
        assert_eq!(
            prepared
                .for_dimension(64)
                .expect("prepared D64 quotient factors")
                .ring_dim,
            64
        );
        let PreparedCoefficientPoint::Lifted { additional, .. } = &prepared.coefficient_point
        else {
            panic!("expected lifted coefficient point");
        };
        assert_eq!(additional.len(), 1);
    }
}
