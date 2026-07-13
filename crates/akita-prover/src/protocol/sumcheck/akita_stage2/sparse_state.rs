use akita_algebra::offset_eq::eq_interval_weights;
use akita_algebra::poly::trim_trailing_zeros;
use akita_field::{AkitaError, FieldCore, FromPrimitiveInt};
use akita_sumcheck::UniPoly;
use akita_types::RelationLayout;

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

    /// Initialize the restricted binary range-check weights from the checked
    /// physical relation layout.
    ///
    /// The initializer derives the complete sorted, disjoint, coalesced support
    /// and padded Boolean domain from the layout compiled from the authenticated
    /// statement; callers cannot substitute a second support or padding
    /// authority. It visits exactly those live coefficients and stores
    /// `rho_bin * eq(r_virt, z)` at address `z`; it never materializes either a
    /// domain-sized support bitmap or a domain-sized equality table. Ordinary
    /// sparse relation weights share the same state so every Stage-2 execution
    /// path combines and binds the two additions through one implementation.
    // Kept internal and deliberately unwired until the schedule descriptor and
    // `rho_bin` transcript event land atomically; this slice exercises it
    // through the nonempty protocol harness below.
    #[allow(dead_code)]
    pub(crate) fn with_negative_binary_support(
        layout: &RelationLayout,
        domain_len: usize,
        relation: Vec<(usize, E)>,
        r_virt: &[E],
        rho_bin: E,
    ) -> Result<Self, AkitaError> {
        let physical_len = layout.physical_witness_field_coeff_len()?;
        let expected_domain_len = physical_len.checked_next_power_of_two().ok_or_else(|| {
            AkitaError::InvalidInput("Stage-2 physical witness domain overflow".into())
        })?;
        if domain_len != expected_domain_len {
            return Err(AkitaError::InvalidSize {
                expected: expected_domain_len,
                actual: domain_len,
            });
        }
        let support = layout.physical_negative_binary_support()?;
        let restricted_eq = Self::restricted_eq_entries(
            domain_len,
            support.iter().map(|span| (span.start(), span.len())),
            r_virt,
            rho_bin,
        )?;
        Self::new(domain_len, relation, restricted_eq)
    }

    fn restricted_eq_entries<I>(
        domain_len: usize,
        runs: I,
        r_virt: &[E],
        rho_bin: E,
    ) -> Result<Vec<(usize, E)>, AkitaError>
    where
        I: Clone + IntoIterator<Item = (usize, usize)>,
    {
        if domain_len == 0 || !domain_len.is_power_of_two() {
            return Err(AkitaError::InvalidInput(
                "sparse Stage-2 domain length must be a nonzero power of two".into(),
            ));
        }
        if r_virt.len() != domain_len.trailing_zeros() as usize {
            return Err(AkitaError::InvalidSize {
                expected: domain_len.trailing_zeros() as usize,
                actual: r_virt.len(),
            });
        }
        let mut run_count = 0usize;
        let mut support_len = 0usize;
        let mut previous_end = None;
        for (start, len) in runs.clone() {
            run_count += 1;
            if len == 0 {
                return Err(AkitaError::InvalidInput(
                    "negative-binary support must omit empty runs".into(),
                ));
            }
            let end = start.checked_add(len).ok_or_else(|| {
                AkitaError::InvalidInput("negative-binary support endpoint overflow".into())
            })?;
            if end > domain_len {
                return Err(AkitaError::InvalidSize {
                    expected: domain_len,
                    actual: end,
                });
            }
            if previous_end.is_some_and(|prior| prior >= start) {
                return Err(AkitaError::InvalidInput(
                    "negative-binary support runs must be sorted, disjoint, and coalesced".into(),
                ));
            }
            support_len = support_len.checked_add(len).ok_or_else(|| {
                AkitaError::InvalidInput("negative-binary support length overflow".into())
            })?;
            previous_end = Some(end);
        }
        if run_count == 0 {
            return Err(AkitaError::InvalidInput(
                "negative-binary Stage-2 support must be nonempty".into(),
            ));
        }

        let mut entries = Vec::new();
        entries.try_reserve_exact(support_len).map_err(|_| {
            AkitaError::InvalidInput("negative-binary support allocation failed".into())
        })?;
        for (start, len) in runs {
            let weights = eq_interval_weights(r_virt, start, len)?;
            for (local, eq_weight) in weights.into_iter().enumerate() {
                let weight = rho_bin * eq_weight;
                if weight != E::zero() {
                    entries.push((start + local, weight));
                }
            }
        }
        Ok(entries)
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
    use akita_algebra::offset_eq::eq_eval_at_index;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::{CanonicalField, Prime128Offset275, Prime128OffsetA7F7};
    use akita_types::layout::{CoeffSpan, RelationSegmentId};
    use akita_types::sis::{
        sis_table_key_for_linf_bound, AjtaiKeyParams, DEFAULT_SIS_SECURITY_BITS,
    };
    use akita_types::{
        validate_compression_catalog, CompressionAlphabet, CompressionCatalogContext,
        CompressionChainSpec, CompressionMapSpec, CompressionSourceId, LevelParams,
        OpeningClaimsLayout, SisModulusFamily,
    };

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

    fn certified_key(d: usize, raw_bound: u128, cols: usize) -> AjtaiKeyParams {
        let table = sis_table_key_for_linf_bound(
            DEFAULT_SIS_SECURITY_BITS,
            SisModulusFamily::Q128,
            d as u32,
            raw_bound,
        )
        .expect("test SIS row");
        AjtaiKeyParams::try_new_with_min_rank(table, cols).expect("certified test key")
    }

    fn compression_chain(
        source: CompressionSourceId,
        source_key: &AjtaiKeyParams,
        alphabets: &[CompressionAlphabet],
    ) -> CompressionChainSpec {
        let mut previous =
            source_key.row_len() * source_key.sis_table_key().ring_dimension as usize;
        let maps = alphabets
            .iter()
            .copied()
            .enumerate()
            .map(|(map, alphabet)| {
                let d = if map == 0 { 64 } else { 32 };
                let depth = match alphabet {
                    CompressionAlphabet::NegativeBinary => {
                        Prime128OffsetA7F7::modulus_bits() as usize
                    }
                    CompressionAlphabet::OpeningBase { log_basis } => {
                        akita_types::sis::num_digits_for_bound(
                            Prime128OffsetA7F7::modulus_bits(),
                            Prime128OffsetA7F7::modulus_bits(),
                            log_basis,
                        )
                    }
                };
                let bound = match alphabet {
                    CompressionAlphabet::NegativeBinary => 1,
                    CompressionAlphabet::OpeningBase { log_basis } => (1u128 << log_basis) - 1,
                };
                let key = certified_key(d, bound, previous * depth / d);
                previous = key.row_len() * d;
                CompressionMapSpec::new(key, alphabet)
            })
            .collect();
        CompressionChainSpec::new(source, 6, maps)
    }

    fn multilayer_layout() -> akita_types::RelationLayout {
        let mut level = LevelParams::params_only(
            SisModulusFamily::Q128,
            64,
            6,
            1,
            1,
            1,
            SparseChallengeConfig::pm1_only(64),
        )
        .with_decomp(1, 1, 1, 1, 0)
        .expect("level");
        level.b_key = certified_key(64, 63, 1);
        level.d_key = certified_key(64, 63, 1);
        let opening = OpeningClaimsLayout::new(2, 1).expect("opening");
        validate_compression_catalog::<Prime128OffsetA7F7>(
            &level,
            CompressionCatalogContext::CoGeneratedLevel { opening: &opening },
            64,
            vec![
                compression_chain(
                    CompressionSourceId::CurrentOuter,
                    &level.b_key,
                    &[
                        CompressionAlphabet::NegativeBinary,
                        CompressionAlphabet::NegativeBinary,
                        CompressionAlphabet::NegativeBinary,
                    ],
                ),
                compression_chain(
                    CompressionSourceId::Opening,
                    &level.d_key,
                    &[
                        CompressionAlphabet::OpeningBase { log_basis: 6 },
                        CompressionAlphabet::NegativeBinary,
                    ],
                ),
            ],
        )
        .expect("catalog")
        .co_generated_relation_layout()
        .expect("relation layout")
        .clone()
    }

    #[test]
    fn checked_constructor_rejects_noncanonical_support() {
        assert!(Stage2SparseState::<F>::new(3, vec![], vec![]).is_err());
        assert!(Stage2SparseState::new(8, vec![(2, f(1)), (1, f(2))], vec![]).is_err());
        assert!(Stage2SparseState::new(8, vec![(2, f(1)), (2, f(2))], vec![]).is_err());
        assert!(Stage2SparseState::new(8, vec![(8, f(1))], vec![]).is_err());
        assert!(Stage2SparseState::new(8, vec![(1, F::zero())], vec![]).is_err());

        let point = [f(2), f(3), f(5)];
        assert!(
            Stage2SparseState::restricted_eq_entries(8, std::iter::empty(), &point, f(7),).is_err()
        );
        assert!(Stage2SparseState::restricted_eq_entries(8, [(1, 0)], &point, f(7)).is_err());
        assert!(
            Stage2SparseState::restricted_eq_entries(8, [(2, 2), (3, 1)], &point, f(7)).is_err()
        );
        assert!(
            Stage2SparseState::restricted_eq_entries(8, [(2, 2), (2, 1)], &point, f(7)).is_err()
        );
        assert!(Stage2SparseState::restricted_eq_entries(8, [(7, 2)], &point, f(7)).is_err());
        assert!(
            Stage2SparseState::restricted_eq_entries(8, [(1, usize::MAX)], &point, f(7),).is_err()
        );
        assert!(Stage2SparseState::restricted_eq_entries(8, [(1, 1)], &point[..2], f(7)).is_err());
    }

    #[test]
    fn projected_multilayer_binary_support_matches_verifier_terminal_formula() {
        let layout = multilayer_layout();
        let support = layout
            .physical_negative_binary_support()
            .expect("projected support");
        assert!(
            support.len() >= 2,
            "opening-base gap must preserve multiple runs"
        );

        // Every negative-binary input layer is included, while the opening-base
        // first H layer is absent. This checks the authenticated alphabet tag,
        // not a hard-coded assumption about map indices.
        for (source, map, is_binary) in [
            (CompressionSourceId::CurrentOuter, 0, true),
            (CompressionSourceId::CurrentOuter, 1, true),
            (CompressionSourceId::CurrentOuter, 2, true),
            (CompressionSourceId::Opening, 0, false),
            (CompressionSourceId::Opening, 1, true),
        ] {
            let span = layout
                .physical_compression_segment_span(RelationSegmentId::CompressionInput {
                    source,
                    map,
                })
                .expect("input span");
            let covered = support.iter().any(|run| {
                run.start() <= span.start() && run.start() + run.len() >= span.start() + span.len()
            });
            assert_eq!(
                covered, is_binary,
                "wrong support membership for {source:?}/{map}"
            );
        }

        let domain_len = layout
            .physical_witness_field_coeff_len()
            .expect("physical witness length")
            .next_power_of_two();
        let num_vars = domain_len.trailing_zeros() as usize;
        let r_virt: Vec<F> = (0..num_vars).map(|i| f(2 * i as u64 + 3)).collect();
        let rho = f(41);
        let relation = Vec::new();
        let mut state = Stage2SparseState::with_negative_binary_support(
            &layout, domain_len, relation, &r_virt, rho,
        )
        .expect("augmented sparse state");
        assert!(Stage2SparseState::with_negative_binary_support(
            &layout,
            domain_len / 2,
            Vec::new(),
            &r_virt,
            rho,
        )
        .is_err());
        let initial_support = state.restricted_eq.len();
        assert!(initial_support > 0);
        assert!(initial_support <= support.iter().map(CoeffSpan::len).sum());

        let mut witness: Vec<F> = (0..domain_len)
            .map(|index| f((index as u64).wrapping_mul(17).wrapping_add(5)))
            .collect();
        let mut dense_weights = vec![F::zero(); domain_len];
        for &(index, weight) in &state.restricted_eq {
            dense_weights[index] = weight;
        }
        let prefix_grid = state
            .two_round_grid(|index| witness[index])
            .expect("nonempty sparse prefix grid");
        assert_eq!(
            prefix_grid.round0_poly(),
            state.round_poly(|index| witness[index])
        );

        let r_stage2: Vec<F> = (0..num_vars).map(|i| f(3 * i as u64 + 11)).collect();
        let mut previous_support = initial_support;
        for (round, &challenge) in r_stage2.iter().enumerate() {
            let round_poly = state.round_poly(|index| witness[index]);
            for t in [F::zero(), F::one(), f(2), f(7)] {
                let expected = witness
                    .chunks_exact(2)
                    .zip(dense_weights.chunks_exact(2))
                    .fold(F::zero(), |acc, (w, weight)| {
                        let w_t = w[0] + t * (w[1] - w[0]);
                        let weight_t = weight[0] + t * (weight[1] - weight[0]);
                        acc + weight_t * w_t * (w_t + F::one())
                    });
                assert_eq!(round_poly.evaluate(&t), expected, "round {round}");
            }
            if round == 0 {
                let mut once_bound = state.clone();
                once_bound.bind(challenge);
                let folded_witness: Vec<F> = witness
                    .chunks_exact(2)
                    .map(|pair| pair[0] + challenge * (pair[1] - pair[0]))
                    .collect();
                assert_eq!(
                    prefix_grid.round1_poly(challenge),
                    once_bound.round_poly(|index| folded_witness[index])
                );
            }
            state.bind(challenge);
            assert!(state.restricted_eq.len() <= previous_support);
            previous_support = state.restricted_eq.len();
            witness = witness
                .chunks_exact(2)
                .map(|pair| pair[0] + challenge * (pair[1] - pair[0]))
                .collect();
            dense_weights = dense_weights
                .chunks_exact(2)
                .map(|pair| pair[0] + challenge * (pair[1] - pair[0]))
                .collect();
        }
        let prover_terminal = state
            .restricted_eq
            .first()
            .map_or(F::zero(), |&(index, value)| {
                assert_eq!(index, 0);
                value
            });
        let verifier_terminal = support.iter().fold(F::zero(), |acc, run| {
            acc + run.range().fold(F::zero(), |run_acc, index| {
                run_acc
                    + rho * eq_eval_at_index(&r_virt, index) * eq_eval_at_index(&r_stage2, index)
            })
        });
        assert_eq!(prover_terminal, verifier_terminal);
        assert_eq!(prover_terminal, dense_weights[0]);
    }

    #[test]
    fn binary_support_initializer_filters_algebraic_zeros_without_rejecting_support() {
        let point = [F::zero(), F::one(), F::zero()];
        let zero_rho =
            Stage2SparseState::restricted_eq_entries(8, [(0, 3), (5, 2)], &point, F::zero())
                .expect("structurally live support with zero challenge");
        assert!(zero_rho.is_empty());

        let entries = Stage2SparseState::restricted_eq_entries(8, [(0, 3), (5, 2)], &point, f(13))
            .expect("degenerate Boolean point");
        assert_eq!(entries, vec![(2, f(13))]);
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
