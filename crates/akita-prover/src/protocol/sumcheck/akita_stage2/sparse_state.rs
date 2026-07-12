use akita_algebra::poly::trim_trailing_zeros;
use akita_field::{AkitaError, FieldCore, FromPrimitiveInt};
use akita_sumcheck::UniPoly;

/// Sparse Stage-2 weights in the current full Boolean address space.
///
/// Entries are canonical: strictly increasing, in range, and nonzero. Relation
/// weights contribute `p(z)w(z)`; restricted-equality weights contribute
/// `p(z)w(z)(w(z)+1)`.
#[derive(Clone)]
pub(crate) struct Stage2SparseState<E: FieldCore> {
    domain_len: usize,
    relation: Vec<(usize, E)>,
    restricted_eq: Vec<(usize, E)>,
}

/// Degree-three bivariate evaluations for the first two sparse rounds.
pub(crate) struct SparseTwoRoundGrid<E: FieldCore> {
    evals: [[E; 4]; 4],
}

impl<E: FieldCore> Stage2SparseState<E> {
    pub(crate) fn empty(domain_len: usize) -> Result<Self, AkitaError> {
        Self::new(domain_len, Vec::new(), Vec::new())
    }

    pub(crate) fn new(
        domain_len: usize,
        relation: Vec<(usize, E)>,
        restricted_eq: Vec<(usize, E)>,
    ) -> Result<Self, AkitaError> {
        if domain_len == 0 || !domain_len.is_power_of_two() {
            return Err(AkitaError::InvalidInput(
                "sparse Stage-2 domain length must be a nonzero power of two".into(),
            ));
        }
        Self::validate_entries(domain_len, &relation, "relation")?;
        Self::validate_entries(domain_len, &restricted_eq, "restricted equality")?;
        Ok(Self {
            domain_len,
            relation,
            restricted_eq,
        })
    }

    fn validate_entries(
        domain_len: usize,
        entries: &[(usize, E)],
        label: &str,
    ) -> Result<(), AkitaError> {
        let mut previous = None;
        for &(index, weight) in entries {
            if index >= domain_len {
                return Err(AkitaError::InvalidSize {
                    expected: domain_len,
                    actual: index.saturating_add(1),
                });
            }
            if previous.is_some_and(|prior| prior >= index) {
                return Err(AkitaError::InvalidInput(format!(
                    "sparse Stage-2 {label} indices must be strictly increasing"
                )));
            }
            if weight == E::zero() {
                return Err(AkitaError::InvalidInput(format!(
                    "sparse Stage-2 {label} support must omit zero weights"
                )));
            }
            previous = Some(index);
        }
        Ok(())
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.relation.is_empty() && self.restricted_eq.is_empty()
    }

    pub(crate) fn round_poly(&self, mut witness_at: impl FnMut(usize) -> E) -> UniPoly<E> {
        if self.is_empty() {
            return UniPoly::from_coeffs(vec![E::zero()]);
        }
        let mut coeffs = [E::zero(); 4];
        self.for_each_active_block::<2>(|pair, relation, restricted_eq| {
            let w0 = witness_at(2 * pair);
            let w1 = witness_at(2 * pair + 1);
            Self::accumulate_pair(&mut coeffs, w0, w1, relation, restricted_eq);
        });
        let mut coeffs = coeffs.to_vec();
        trim_trailing_zeros(&mut coeffs);
        UniPoly::from_coeffs(coeffs)
    }

    fn accumulate_pair(coeffs: &mut [E; 4], w0: E, w1: E, relation: [E; 2], restricted_eq: [E; 2]) {
        let dw = w1 - w0;
        let relation_dp = relation[1] - relation[0];
        coeffs[0] += relation[0] * w0;
        coeffs[1] += relation[0] * dw + relation_dp * w0;
        coeffs[2] += relation_dp * dw;

        let restricted_dp = restricted_eq[1] - restricted_eq[0];
        let q0 = w0 * (w0 + E::one());
        let q1 = dw * (w0 + w0 + E::one());
        let q2 = dw * dw;
        coeffs[0] += restricted_eq[0] * q0;
        coeffs[1] += restricted_eq[0] * q1 + restricted_dp * q0;
        coeffs[2] += restricted_eq[0] * q2 + restricted_dp * q1;
        coeffs[3] += restricted_dp * q2;
    }

    pub(crate) fn bind(&mut self, challenge: E) {
        debug_assert!(
            self.domain_len >= 2,
            "cannot bind an exhausted sparse Stage-2 state"
        );
        if self.is_empty() {
            self.domain_len /= 2;
            return;
        }
        Self::bind_entries(&mut self.relation, challenge);
        Self::bind_entries(&mut self.restricted_eq, challenge);
        self.domain_len /= 2;
    }

    fn bind_entries(entries: &mut Vec<(usize, E)>, challenge: E) {
        let mut bound = Vec::with_capacity(entries.len().div_ceil(2));
        let mut cursor = 0;
        while cursor < entries.len() {
            let parent = entries[cursor].0 / 2;
            let mut endpoints = [E::zero(); 2];
            while cursor < entries.len() && entries[cursor].0 / 2 == parent {
                endpoints[entries[cursor].0 & 1] = entries[cursor].1;
                cursor += 1;
            }
            let value = endpoints[0] + challenge * (endpoints[1] - endpoints[0]);
            if value != E::zero() {
                bound.push((parent, value));
            }
        }
        *entries = bound;
    }

    pub(crate) fn two_round_grid(
        &self,
        mut witness_at: impl FnMut(usize) -> E,
    ) -> Result<SparseTwoRoundGrid<E>, AkitaError>
    where
        E: FromPrimitiveInt,
    {
        if self.domain_len < 4 {
            return Err(AkitaError::InvalidInput(
                "sparse two-round grid requires at least two remaining variables".into(),
            ));
        }
        let points = [E::zero(), E::one(), E::from_u64(2), E::from_u64(3)];
        let mut evals = [[E::zero(); 4]; 4];
        self.for_each_active_block::<4>(|quad, relation, restricted_eq| {
            let witness = std::array::from_fn(|corner| witness_at(4 * quad + corner));
            for (x_index, &x) in points.iter().enumerate() {
                for (y_index, &y) in points.iter().enumerate() {
                    let w = bilinear_eval(witness, x, y);
                    let relation_weight = bilinear_eval(relation, x, y);
                    let restricted_weight = bilinear_eval(restricted_eq, x, y);
                    evals[x_index][y_index] +=
                        relation_weight * w + restricted_weight * w * (w + E::one());
                }
            }
        });
        Ok(SparseTwoRoundGrid { evals })
    }

    fn for_each_active_block<const N: usize>(&self, mut visit: impl FnMut(usize, [E; N], [E; N])) {
        let mut relation_cursor = 0;
        let mut restricted_cursor = 0;
        while relation_cursor < self.relation.len() || restricted_cursor < self.restricted_eq.len()
        {
            let relation_block = self
                .relation
                .get(relation_cursor)
                .map_or(usize::MAX, |entry| entry.0 / N);
            let restricted_block = self
                .restricted_eq
                .get(restricted_cursor)
                .map_or(usize::MAX, |entry| entry.0 / N);
            let block = relation_block.min(restricted_block);
            let mut relation = [E::zero(); N];
            while relation_cursor < self.relation.len()
                && self.relation[relation_cursor].0 / N == block
            {
                relation[self.relation[relation_cursor].0 % N] = self.relation[relation_cursor].1;
                relation_cursor += 1;
            }
            let mut restricted_eq = [E::zero(); N];
            while restricted_cursor < self.restricted_eq.len()
                && self.restricted_eq[restricted_cursor].0 / N == block
            {
                restricted_eq[self.restricted_eq[restricted_cursor].0 % N] =
                    self.restricted_eq[restricted_cursor].1;
                restricted_cursor += 1;
            }
            visit(block, relation, restricted_eq);
        }
    }
}

impl<E: FieldCore + FromPrimitiveInt> SparseTwoRoundGrid<E> {
    pub(crate) fn round0_poly(&self) -> UniPoly<E> {
        let evals: [E; 4] = std::array::from_fn(|x| self.evals[x][0] + self.evals[x][1]);
        UniPoly::from_evals(&evals)
    }

    pub(crate) fn round1_poly(&self, first_challenge: E) -> UniPoly<E> {
        let evals: [E; 4] = std::array::from_fn(|y| {
            let x_evals: [E; 4] = std::array::from_fn(|x| self.evals[x][y]);
            UniPoly::from_evals(&x_evals).evaluate(&first_challenge)
        });
        UniPoly::from_evals(&evals)
    }
}

fn bilinear_eval<E: FieldCore>(corners: [E; 4], x: E, y: E) -> E {
    let low = corners[0] + x * (corners[1] - corners[0]);
    let high = corners[2] + x * (corners[3] - corners[2]);
    low + y * (high - low)
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Prime128Offset275;

    type F = Prime128Offset275;

    fn f(value: u64) -> F {
        F::from_u64(value)
    }

    fn dense_weights(entries: &[(usize, F)], len: usize) -> Vec<F> {
        let mut dense = vec![F::zero(); len];
        for &(index, weight) in entries {
            dense[index] = weight;
        }
        dense
    }

    #[test]
    fn checked_constructor_rejects_noncanonical_support() {
        assert!(Stage2SparseState::<F>::new(3, vec![], vec![]).is_err());
        assert!(Stage2SparseState::new(8, vec![(2, f(1)), (1, f(2))], vec![]).is_err());
        assert!(Stage2SparseState::new(8, vec![(2, f(1)), (2, f(2))], vec![]).is_err());
        assert!(Stage2SparseState::new(8, vec![(8, f(1))], vec![]).is_err());
        assert!(Stage2SparseState::new(8, vec![(1, F::zero())], vec![]).is_err());
    }

    #[test]
    fn sparse_round_poly_matches_dense_oracle_and_support_never_grows() {
        let relation = vec![(0, f(2)), (3, f(5)), (4, f(7))];
        let restricted = vec![(1, f(11)), (2, f(13)), (7, f(17))];
        let witness = [f(3), f(6), f(2), f(9), f(4), f(8), f(5), f(1)];
        let relation_dense = dense_weights(&relation, 8);
        let restricted_dense = dense_weights(&restricted, 8);
        let mut state = Stage2SparseState::new(8, relation, restricted).expect("state");
        let poly = state.round_poly(|index| witness[index]);

        for t in [f(0), f(1), f(2), f(7)] {
            let mut expected = F::zero();
            for pair in 0..4 {
                let w = witness[2 * pair] + t * (witness[2 * pair + 1] - witness[2 * pair]);
                let relation_weight = relation_dense[2 * pair]
                    + t * (relation_dense[2 * pair + 1] - relation_dense[2 * pair]);
                let restricted_weight = restricted_dense[2 * pair]
                    + t * (restricted_dense[2 * pair + 1] - restricted_dense[2 * pair]);
                expected += relation_weight * w + restricted_weight * w * (w + F::one());
            }
            assert_eq!(poly.evaluate(&t), expected);
        }

        let mut previous_support = state.relation.len() + state.restricted_eq.len();
        for challenge in [f(19), f(23), f(29)] {
            state.bind(challenge);
            let support = state.relation.len() + state.restricted_eq.len();
            assert!(support <= previous_support);
            previous_support = support;
        }
        assert_eq!(state.domain_len, 1);
    }

    #[test]
    fn sparse_two_round_grid_axis_order_matches_dense_oracle() {
        let relation = vec![(0, f(2)), (3, f(5)), (4, f(7))];
        let restricted = vec![(1, f(11)), (2, f(13)), (7, f(17))];
        let witness = [f(3), f(6), f(2), f(9), f(4), f(8), f(5), f(1)];
        let relation_dense = dense_weights(&relation, 8);
        let restricted_dense = dense_weights(&restricted, 8);
        let state = Stage2SparseState::new(8, relation, restricted).expect("state");
        let grid = state.two_round_grid(|index| witness[index]).expect("grid");
        let points = [f(0), f(1), f(2), f(3)];

        for (x_index, &x) in points.iter().enumerate() {
            for (y_index, &y) in points.iter().enumerate() {
                let mut expected = F::zero();
                for quad in 0..2 {
                    let witness_quad = std::array::from_fn(|corner| witness[4 * quad + corner]);
                    let relation_quad =
                        std::array::from_fn(|corner| relation_dense[4 * quad + corner]);
                    let restricted_quad =
                        std::array::from_fn(|corner| restricted_dense[4 * quad + corner]);
                    let w = bilinear_eval(witness_quad, x, y);
                    expected += bilinear_eval(relation_quad, x, y) * w
                        + bilinear_eval(restricted_quad, x, y) * w * (w + F::one());
                }
                assert_eq!(grid.evals[x_index][y_index], expected);
            }
        }

        let r0 = f(23);
        let round0 = state.round_poly(|index| witness[index]);
        assert_eq!(grid.round0_poly(), round0);

        let mut bound_state = state.clone();
        bound_state.bind(r0);
        let folded_witness: [F; 4] = std::array::from_fn(|pair| {
            witness[2 * pair] + r0 * (witness[2 * pair + 1] - witness[2 * pair])
        });
        assert_eq!(
            grid.round1_poly(r0),
            bound_state.round_poly(|index| folded_witness[index])
        );
        for t in points {
            let mut expected = F::zero();
            for quad in 0..2 {
                let witness_quad = std::array::from_fn(|corner| witness[4 * quad + corner]);
                let relation_quad = std::array::from_fn(|corner| relation_dense[4 * quad + corner]);
                let restricted_quad =
                    std::array::from_fn(|corner| restricted_dense[4 * quad + corner]);
                let w = bilinear_eval(witness_quad, r0, t);
                expected += bilinear_eval(relation_quad, r0, t) * w
                    + bilinear_eval(restricted_quad, r0, t) * w * (w + F::one());
            }
            assert_eq!(grid.round1_poly(r0).evaluate(&t), expected);
        }
    }

    #[test]
    fn empty_state_has_canonical_zero_contributions() {
        let mut state = Stage2SparseState::<F>::empty(8).expect("empty state");
        assert_eq!(state.round_poly(|_| f(9)).coeffs, vec![F::zero()]);
        let grid = state.two_round_grid(|_| f(9)).expect("empty grid");
        assert_eq!(grid.round0_poly().coeffs, vec![F::zero()]);
        assert_eq!(grid.round1_poly(f(5)).coeffs, vec![F::zero()]);
        state.bind(f(7));
        assert_eq!(state.domain_len, 4);
        assert!(state.relation.is_empty());
        assert!(state.restricted_eq.is_empty());
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "cannot bind an exhausted sparse Stage-2 state")]
    fn exhausted_state_rejects_an_extra_bind() {
        let mut state = Stage2SparseState::<F>::empty(1).expect("empty state");
        state.bind(f(7));
    }
}
