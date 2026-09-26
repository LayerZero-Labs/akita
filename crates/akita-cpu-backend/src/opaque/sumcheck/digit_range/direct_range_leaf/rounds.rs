use super::*;
use jolt_poly::OmittedConstantPoly;

impl<E: Field + Ring + Unreduced> LowBasisRangeCheckProver<E> {
    pub(super) fn compute_current_round_eq_poly_from_state(&mut self) -> OmittedConstantPoly<E> {
        let use_octet_prefix = self.using_octet_prefix();
        let use_prefix_x_round = !use_octet_prefix && self.use_prefix_x_round();
        let use_sparse_x_y_round = !use_octet_prefix && self.use_sparse_x_y_round();
        let rounds_completed = self.rounds_completed;
        let phase = if use_octet_prefix {
            "octet-prefix"
        } else if use_prefix_x_round {
            "live-prefix"
        } else if use_sparse_x_y_round {
            "sparse-low-variables"
        } else {
            "dense"
        };
        let _span = tracing::info_span!(
            "digit_range_direct_leaf_round",
            round = rounds_completed,
            phase
        )
        .entered();
        if use_octet_prefix {
            return self.compute_octet_prefix_round();
        }
        let LowBasisRangeImageStorage::Materialized(range_image) = &self.range_image else {
            unreachable!("compact storage belongs to the octet prefix");
        };
        if use_prefix_x_round || use_sparse_x_y_round {
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
        } else {
            compute_range_round_polynomial_from_range_image(&self.split_eq, &self.range_poly, |j| {
                (range_image[2 * j], range_image[2 * j + 1])
            })
        }
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
        if self.using_octet_prefix() {
            self.ingest_octet_prefix_challenge(r);
        } else {
            self.ingest_generic_challenge(r);
        }
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
    fn ingest_generic_challenge(&mut self, r: E) {
        self.split_eq.bind(r);
        let use_prefix_x_round = self.use_prefix_x_round();
        let use_live_prefix = use_prefix_x_round || self.use_sparse_x_y_round();
        let fuse_next_live_prefix = use_live_prefix && self.next_round_uses_live_prefix();
        let LowBasisRangeImageStorage::Materialized(range_image) = &mut self.range_image else {
            unreachable!("compact storage belongs to the octet prefix");
        };
        let mut range_image = std::mem::take(range_image);
        self.range_image = {
            if use_live_prefix {
                if fuse_next_live_prefix {
                    let range_image = range_image.as_slice();
                    let (next_range_image, round_poly) = self.fuse_live_prefix_and_compute_round(
                        range_image.len().div_ceil(2),
                        |_| {
                            move |entry| {
                                fold_prefix_pair_with_zero_padding(range_image, 2 * entry, r)
                            }
                        },
                    );
                    self.cached_round_poly = Some(round_poly);
                    LowBasisRangeImageStorage::Materialized(next_range_image)
                } else {
                    LowBasisRangeImageStorage::Materialized(Self::fold_live_prefix(&range_image, r))
                }
            } else {
                fold_evals_in_place(&mut range_image, r);
                LowBasisRangeImageStorage::Materialized(range_image)
            }
        };
    }
}
