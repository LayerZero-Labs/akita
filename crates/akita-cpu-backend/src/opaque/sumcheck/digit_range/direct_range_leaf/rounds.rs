use super::*;
use jolt_poly::OmittedConstantPoly;

impl<E: Field + Ring + Unreduced> LowBasisRangeCheckProver<E> {
    pub(super) fn compute_current_round_eq_poly_from_state(&mut self) -> OmittedConstantPoly<E> {
        let kernel = self.round_kernel(self.rounds_completed);
        let rounds_completed = self.rounds_completed;
        let phase = match kernel {
            RoundKernel::OctetPrefix => "octet-prefix",
            RoundKernel::LivePrefix => "live-prefix",
            RoundKernel::Dense => "dense",
        };
        let _span = tracing::info_span!(
            "digit_range_direct_leaf_round",
            round = rounds_completed,
            phase
        )
        .entered();
        let mut storage = std::mem::replace(
            &mut self.range_image,
            LowBasisRangeImageStorage::Materialized(Vec::new()),
        );
        let poly = match (&mut storage, kernel) {
            (LowBasisRangeImageStorage::OctetPrefix(prefix), _) => {
                self.compute_octet_prefix_round(prefix)
            }
            (LowBasisRangeImageStorage::Materialized(range_image), RoundKernel::LivePrefix) => {
                let range_image = range_image.as_slice();
                let precomputation = &self.range_poly;
                let sums = self.compute_round_live_prefix(range_image.len().div_ceil(2), |_| {
                    move |pair, weight, sums| {
                        let left = range_image[2 * pair];
                        let right = range_image
                            .get(2 * pair + 1)
                            .copied()
                            .unwrap_or_else(E::zero);
                        precomputation.accumulate_entry_terms(sums, left, right - left, weight);
                    }
                });
                precomputation.round_poly_from_sums(&sums, LinearSum::Taylor)
            }
            (LowBasisRangeImageStorage::Materialized(range_image), _) => {
                compute_range_round_polynomial_from_range_image(
                    &self.split_eq,
                    &self.range_poly,
                    |j| (range_image[2 * j], range_image[2 * j + 1]),
                )
            }
        };
        self.range_image = storage;
        poly
    }
}

impl<E: Field + Ring + Unreduced + Fold> EqFactoredSumcheckInstanceProver<E>
    for LowBasisRangeCheckProver<E>
{
    fn num_rounds(&self) -> usize {
        self.num_vars
    }

    fn degree_bound(&self) -> usize {
        self.basis / 2
    }

    fn input_claim(&self) -> E {
        E::zero()
    }

    fn current_tau(&self) -> E {
        self.split_eq.current_tau()
    }

    fn compute_round_eq_factored(&mut self, round: usize) -> OmittedConstantPoly<E> {
        debug_assert_eq!(round, self.rounds_completed);
        if let Some(poly) = self.cached_round_poly.take() {
            poly
        } else {
            self.compute_current_round_eq_poly_from_state()
        }
    }

    fn ingest_challenge(&mut self, round: usize, r: E) {
        debug_assert_eq!(round, self.rounds_completed);
        let _span = tracing::info_span!(
            "digit_range_direct_leaf_fold",
            round = self.rounds_completed
        )
        .entered();
        let current_kernel = self.round_kernel(self.rounds_completed);
        let next_kernel = self.round_kernel(self.rounds_completed + 1);
        let storage = std::mem::replace(
            &mut self.range_image,
            LowBasisRangeImageStorage::Materialized(Vec::new()),
        );
        self.range_image = match storage {
            LowBasisRangeImageStorage::OctetPrefix(mut prefix) => {
                match self.ingest_octet_prefix_challenge(&mut prefix, r) {
                    Some(table) => LowBasisRangeImageStorage::Materialized(table),
                    None => LowBasisRangeImageStorage::OctetPrefix(prefix),
                }
            }
            LowBasisRangeImageStorage::Materialized(table) => {
                LowBasisRangeImageStorage::Materialized(self.ingest_generic_challenge(
                    table,
                    r,
                    current_kernel,
                    next_kernel,
                ))
            }
        };
        if self.in_x_phase() {
            self.live_x_cols = self.live_x_cols.div_ceil(2);
        }
        self.rounds_completed += 1;
        if self.rounds_completed < self.num_vars {
            if self.cached_round_poly.is_none() {
                self.cached_round_poly = Some(self.compute_current_round_eq_poly_from_state());
            }
        } else {
            self.cached_round_poly = None;
        }
    }
}

impl<E: Field + Ring + Unreduced + Fold> LowBasisRangeCheckProver<E> {
    fn ingest_generic_challenge(
        &mut self,
        mut range_image: Vec<E>,
        r: E,
        current: RoundKernel,
        next: RoundKernel,
    ) -> Vec<E> {
        self.split_eq.bind(r);
        let fuse_next_live_prefix =
            current == RoundKernel::LivePrefix && next == RoundKernel::LivePrefix;
        if fuse_next_live_prefix {
            let range_image = range_image.as_slice();
            let (next_range_image, round_poly) = self
                .fuse_live_prefix_and_compute_round(range_image.len().div_ceil(2), |_| {
                    move |entry| fold_prefix_pair_with_zero_padding(range_image, 2 * entry, r)
                });
            self.cached_round_poly = Some(round_poly);
            next_range_image
        } else if current == RoundKernel::LivePrefix {
            Self::fold_live_prefix(&range_image, r)
        } else {
            fold_evals_in_place(&mut range_image, r);
            range_image
        }
    }
}
