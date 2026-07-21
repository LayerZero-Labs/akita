//! Test-first contract for the canonical flat relation point.
//!
//! This module stays test-only until the verifier's current uniform and
//! lane-factored formulas have differential parity through every contribution.
//! The production cutover will make this geometry the single point authority;
//! landing it as unused production state would instead create a third path.

use akita_algebra::{offset_eq::OffsetEqWindow, poly::multilinear_eval, ring::scalar_powers};
use akita_field::{AkitaError, FieldCore};
use akita_types::{
    checked_opening_source_index, opening_domain_len, validate_role_dims, CommitmentRingDims,
    RingRole,
};
use std::sync::Arc;

struct PreparedRolePoint<E: FieldCore> {
    ring_dim: usize,
    lane_powers: Arc<[E]>,
}

/// Checked factorization of one flat Stage-2 relation point.
///
/// `coeff_count` is the low-address block shared by the relation roles and the
/// outgoing witness representation. The remaining point addresses relation
/// lanes followed by semantic witness columns. Role-native setup columns split
/// one A-role witness column into `d_a / d_role` subcolumns.
pub(super) struct PreparedRelationPoint<E: FieldCore> {
    coeff_count: usize,
    coeff_eval: E,
    equality_window: OffsetEqWindow<E>,
    role_dims: CommitmentRingDims,
    outgoing_ring_dim: usize,
    opening_source_len: usize,
    inner: Arc<PreparedRolePoint<E>>,
    outer: Arc<PreparedRolePoint<E>>,
    opening: Arc<PreparedRolePoint<E>>,
}

impl<E: FieldCore> PreparedRelationPoint<E> {
    pub(super) fn new(
        point: &[E],
        alpha: E,
        role_dims: CommitmentRingDims,
        outgoing_ring_dim: usize,
        opening_source_len: usize,
    ) -> Result<Self, AkitaError> {
        validate_role_dims(role_dims)?;
        if outgoing_ring_dim == 0 || !outgoing_ring_dim.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "outgoing witness ring dimension must be a non-zero power of two".into(),
            ));
        }

        let coeff_count = role_dims.common_relation_witness_coeff_count(outgoing_ring_dim);
        if coeff_count == 0
            || !coeff_count.is_power_of_two()
            || !role_dims.d_a().is_multiple_of(coeff_count)
            || !role_dims.d_b().is_multiple_of(coeff_count)
            || !role_dims.d_d().is_multiple_of(coeff_count)
            || !outgoing_ring_dim.is_multiple_of(coeff_count)
        {
            return Err(AkitaError::InvalidSetup(
                "relation and outgoing witness do not admit a common coefficient block".into(),
            ));
        }

        let field_len = opening_domain_len(opening_source_len)?
            .checked_mul(outgoing_ring_dim)
            .ok_or_else(|| AkitaError::InvalidSetup("relation point domain overflow".into()))?;
        if !field_len.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "relation point domain must be a power of two".into(),
            ));
        }
        let expected = field_len.trailing_zeros() as usize;
        if point.len() != expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: point.len(),
            });
        }

        let coeff_bits = coeff_count.trailing_zeros() as usize;
        let coeff_point = point.get(..coeff_bits).ok_or(AkitaError::InvalidProof)?;
        let lane_and_column_point = point.get(coeff_bits..).ok_or(AkitaError::InvalidProof)?;
        let coeff_powers = scalar_powers(alpha, coeff_count);
        let coeff_eval = multilinear_eval(&coeff_powers, coeff_point)?;
        let equality_window = OffsetEqWindow::new(lane_and_column_point)?;

        let prepare_role = |ring_dim: usize| -> Result<Arc<PreparedRolePoint<E>>, AkitaError> {
            let lane_count = ring_dim
                .checked_div(coeff_count)
                .filter(|&count| count != 0)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("invalid relation role lane count".into())
                })?;
            let powers = scalar_powers(alpha, ring_dim);
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

        Ok(Self {
            coeff_count,
            coeff_eval,
            equality_window,
            role_dims,
            outgoing_ring_dim,
            opening_source_len,
            inner,
            outer,
            opening,
        })
    }

    pub(super) const fn coeff_count(&self) -> usize {
        self.coeff_count
    }

    pub(super) fn coeff_eval(&self) -> E {
        self.coeff_eval
    }

    /// Evaluate the high-address factor for one role-native setup column.
    pub(super) fn role_column_weight(
        &self,
        witness_column: usize,
        role: RingRole,
        role_subcolumn: usize,
    ) -> Result<E, AkitaError> {
        let prepared = self.role(role);
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
        let last_physical = physical_end
            .checked_sub(1)
            .ok_or(AkitaError::InvalidProof)?;
        checked_opening_source_index(
            self.opening_source_len,
            last_physical / self.outgoing_ring_dim,
        )?;
        if !physical_start.is_multiple_of(self.coeff_count) {
            return Err(AkitaError::InvalidProof);
        }
        let lane_start = physical_start / self.coeff_count;

        prepared.lane_powers.iter().copied().enumerate().try_fold(
            E::zero(),
            |evaluation, (lane, alpha_power)| {
                let address = lane_start.checked_add(lane).ok_or_else(|| {
                    AkitaError::InvalidSetup("relation lane address overflow".into())
                })?;
                Ok(evaluation + self.equality_window.eval(address) * alpha_power)
            },
        )
    }

    fn role(&self, role: RingRole) -> &PreparedRolePoint<E> {
        match role {
            RingRole::Inner => &self.inner,
            RingRole::Outer => &self.outer,
            RingRole::Opening => &self.opening,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Prime128OffsetA7F7;

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
        let opening_source_len = 9;
        let field_len = opening_domain_len(opening_source_len).unwrap() * outgoing_ring_dim;
        let point = point_for(field_len);
        let prepared = PreparedRelationPoint::new(
            &point,
            alpha,
            role_dims,
            outgoing_ring_dim,
            opening_source_len,
        )
        .unwrap();

        assert_eq!(
            prepared.coeff_count(),
            role_dims.common_relation_witness_coeff_count(outgoing_ring_dim)
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
                    let got = prepared.coeff_eval()
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
                64,
            ),
            (
                CommitmentRingDims {
                    inner: 128,
                    outer: 64,
                    opening: 32,
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
        let point = point_for(2048);
        assert!(matches!(
            PreparedRelationPoint::new(&point, F::one(), role_dims, 0, 9),
            Err(AkitaError::InvalidSetup(_))
        ));
        assert!(matches!(
            PreparedRelationPoint::new(&point[..point.len() - 1], F::one(), role_dims, 128, 9),
            Err(AkitaError::InvalidSize { .. })
        ));
        let non_nested = CommitmentRingDims {
            inner: 128,
            outer: 32,
            opening: 64,
        };
        assert!(matches!(
            PreparedRelationPoint::new(&point, F::one(), non_nested, 128, 9),
            Err(AkitaError::InvalidSetup(_))
        ));
    }

    #[test]
    fn prepared_relation_point_rejects_out_of_range_witness_columns() {
        let role_dims = CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 32,
        };
        let opening_source_len = 9;
        let outgoing_ring_dim = 32;
        let field_len = opening_domain_len(opening_source_len).unwrap() * outgoing_ring_dim;
        let point = point_for(field_len);
        let prepared = PreparedRelationPoint::new(
            &point,
            F::from_u64(7),
            role_dims,
            outgoing_ring_dim,
            opening_source_len,
        )
        .unwrap();
        assert!(matches!(
            prepared.role_column_weight(3, RingRole::Inner, 0),
            Err(AkitaError::InvalidInput(_))
        ));
        assert!(matches!(
            prepared.role_column_weight(0, RingRole::Outer, 2),
            Err(AkitaError::InvalidProof)
        ));
    }
}
