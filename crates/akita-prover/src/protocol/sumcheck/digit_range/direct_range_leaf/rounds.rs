use super::*;

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps> LowBasisRangeCheckProver<E> {
    pub(super) fn compute_current_round_eq_poly_from_state(&mut self) -> EqFactoredUniPoly<E> {
        let use_two_round_prefix = self.using_two_round_prefix();
        let use_prefix_x_round = !use_two_round_prefix && self.use_prefix_x_round();
        let use_sparse_x_y_round = !use_two_round_prefix && self.use_sparse_x_y_round();
        let rounds_completed = self.rounds_completed;
        let phase = if use_two_round_prefix || use_prefix_x_round {
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
        let poly = if use_two_round_prefix {
            let prefix = self.ensure_initial_round_prefix();
            if rounds_completed == 0 {
                prefix.skip_state.reconstruct_round0_eq_poly()
            } else {
                let r0 = prefix
                    .first_challenge
                    .expect("round 1 prefix polynomial requested before ingesting round 0");
                prefix.skip_state.reconstruct_round1_eq_poly(r0)
            }
        } else if self.split_eq.current_scalar().is_zero() {
            EqFactoredUniPoly::from_q_coeffs(vec![E::zero()])
        } else {
            match &self.range_image {
                LowBasisRangeImageStorage::Compact(compact_range_image) => {
                    if use_prefix_x_round {
                        self.compute_round_compact_prefix_x(compact_range_image)
                    } else if use_sparse_x_y_round {
                        self.compute_round_compact_sparse_x_y(compact_range_image)
                    } else {
                        compute_range_round_polynomial_from_compact_image(
                            &self.split_eq,
                            compact_range_image,
                            &self.polynomial_precomputation,
                        )
                    }
                }
                LowBasisRangeImageStorage::Materialized(range_image) => {
                    if use_prefix_x_round {
                        self.compute_round_full_prefix_x(range_image)
                    } else if use_sparse_x_y_round {
                        self.compute_round_full_sparse_x_y(range_image)
                    } else {
                        compute_range_round_polynomial_from_range_image(
                            &self.split_eq,
                            &self.polynomial_precomputation,
                            |j| (range_image[2 * j], range_image[2 * j + 1]),
                        )
                    }
                }
            }
        };

        poly
    }

    #[tracing::instrument(
        skip_all,
        name = "LowBasisRangeCheckProver::fold_compact_range_image_to_full"
    )]
    pub(super) fn fold_compact_range_image_to_full<V: CompactRangeImageValue>(
        compact_range_image: &[V],
        fold_lut: &CompactPairFoldLut<E>,
    ) -> Vec<E> {
        cfg_into_iter!(0..compact_range_image.len() / 2)
            .map(|j| {
                fold_lut.fold(
                    compact_range_image[2 * j].range_image_value(),
                    compact_range_image[2 * j + 1].range_image_value(),
                )
            })
            .collect()
    }
}

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps + HasOptimizedFold>
    EqFactoredSumcheckInstanceProver<E> for LowBasisRangeCheckProver<E>
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

    fn current_linear_factor_evals(&self) -> (E, E) {
        self.split_eq.linear_factor_evals()
    }

    fn compute_round_eq_factored(&mut self, round: usize) -> EqFactoredUniPoly<E> {
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
        if self.using_two_round_prefix() {
            let rounds_completed = self.rounds_completed;
            self.split_eq.bind(r);
            if rounds_completed == 0 {
                self.ensure_initial_round_prefix().first_challenge = Some(r);
            } else {
                let r0 = {
                    let prefix = self.ensure_initial_round_prefix();
                    prefix
                        .first_challenge
                        .expect("round 1 ingest requires the round 0 challenge")
                };
                let y_len = match &self.range_image {
                    LowBasisRangeImageStorage::Compact(digit_witness) => {
                        digit_witness.len() / self.live_x_cols
                    }
                    LowBasisRangeImageStorage::Materialized(_) => {
                        panic!("two-round prefix expected compact table")
                    }
                };
                if self.defers_binary_range_image_through_third_round() {
                    self.ensure_initial_round_prefix().second_challenge = Some(r);
                    let round_poly = match &self.range_image {
                        LowBasisRangeImageStorage::Compact(compact_range_image) => self
                            .compute_binary_range_image_third_round_from_compact_octets(
                                compact_range_image,
                                r0,
                                r,
                            ),
                        LowBasisRangeImageStorage::Materialized(_) => {
                            unreachable!(
                                "three-round binary range-image deferral requires compact storage"
                            )
                        }
                    };
                    self.cached_round_poly = Some(round_poly);
                } else {
                    self.range_image = match std::mem::replace(
                        &mut self.range_image,
                        LowBasisRangeImageStorage::Materialized(Vec::new()),
                    ) {
                        LowBasisRangeImageStorage::Compact(compact_range_image) => {
                            if self.ring_bits() > 2 {
                                let (range_image, round_poly) = self
                                    .fuse_compact_to_round2_and_compute_round(
                                        &compact_range_image,
                                        r0,
                                        r,
                                    );
                                self.cached_round_poly = Some(round_poly);
                                LowBasisRangeImageStorage::Materialized(range_image)
                            } else {
                                let range_image = Self::fold_compact_range_image_to_round2(
                                    &compact_range_image,
                                    self.live_x_cols,
                                    y_len,
                                    r0,
                                    r,
                                );
                                LowBasisRangeImageStorage::Materialized(range_image)
                            }
                        }
                        LowBasisRangeImageStorage::Materialized(_) => {
                            unreachable!("two-round prefix should hold compact table")
                        }
                    };
                }
            }
            self.rounds_completed += 1;
            if self.rounds_completed < self.num_vars {
                if self.cached_round_poly.is_none() {
                    self.cached_round_poly = Some(self.compute_current_round_eq_poly_from_state());
                }
            } else {
                self.cached_round_poly = None;
            }
            return;
        }

        if self.awaiting_binary_range_image_third_challenge() {
            let (r0, r1) = {
                let prefix = self.ensure_initial_round_prefix();
                (
                    prefix
                        .first_challenge
                        .expect("binary range-image transition requires the first challenge"),
                    prefix
                        .second_challenge
                        .expect("binary range-image transition requires the second challenge"),
                )
            };
            self.split_eq.bind(r);
            let y_len = match &self.range_image {
                LowBasisRangeImageStorage::Compact(digit_witness) => {
                    digit_witness.len() / self.live_x_cols
                }
                LowBasisRangeImageStorage::Materialized(_) => {
                    unreachable!(
                        "three-round binary range-image transition requires compact storage"
                    )
                }
            };
            self.range_image = match std::mem::replace(
                &mut self.range_image,
                LowBasisRangeImageStorage::Materialized(Vec::new()),
            ) {
                LowBasisRangeImageStorage::Compact(compact_range_image) => {
                    LowBasisRangeImageStorage::Materialized(
                        Self::materialize_binary_range_image_after_third_round(
                            &compact_range_image,
                            self.live_x_cols,
                            y_len,
                            r0,
                            r1,
                            r,
                        ),
                    )
                }
                LowBasisRangeImageStorage::Materialized(_) => unreachable!(),
            };
            self.rounds_completed += 1;
            if self.rounds_completed < self.num_vars {
                self.cached_round_poly = Some(self.compute_current_round_eq_poly_from_state());
            } else {
                self.cached_round_poly = None;
            }
            return;
        }

        self.split_eq.bind(r);
        let use_prefix_x_round = self.use_prefix_x_round();
        let use_sparse_x_y_round = self.use_sparse_x_y_round();
        let fuse_next_full_prefix_x =
            use_prefix_x_round && self.next_use_prefix_x_round_after_current();
        let fuse_next_sparse_x_y =
            use_sparse_x_y_round && self.next_use_sparse_x_y_round_after_current();
        let y_len = match &self.range_image {
            LowBasisRangeImageStorage::Compact(digit_witness) => {
                digit_witness.len() / self.live_x_cols
            }
            LowBasisRangeImageStorage::Materialized(range_image) => {
                range_image.len() / self.live_x_cols
            }
        };

        self.range_image = match std::mem::replace(
            &mut self.range_image,
            LowBasisRangeImageStorage::Materialized(Vec::new()),
        ) {
            LowBasisRangeImageStorage::Compact(compact_range_image) => {
                let fold_lut = Self::build_range_image_fold_lut(self.basis, r);
                let range_image = if use_prefix_x_round {
                    Self::fold_compact_range_image_prefix_x(
                        &compact_range_image,
                        self.live_x_cols,
                        y_len,
                        &fold_lut,
                    )
                } else {
                    Self::fold_compact_range_image_to_full(&compact_range_image, &fold_lut)
                };
                LowBasisRangeImageStorage::Materialized(range_image)
            }
            LowBasisRangeImageStorage::Materialized(range_image) => {
                if use_prefix_x_round {
                    if fuse_next_full_prefix_x {
                        let (next_range_image, round_poly) =
                            self.fuse_full_prefix_x_and_compute_round(&range_image, r);
                        self.cached_round_poly = Some(round_poly);
                        LowBasisRangeImageStorage::Materialized(next_range_image)
                    } else {
                        let next_range_image = Self::fold_range_image_prefix_x(
                            &range_image,
                            self.live_x_cols,
                            y_len,
                            r,
                        );
                        LowBasisRangeImageStorage::Materialized(next_range_image)
                    }
                } else if use_sparse_x_y_round {
                    if fuse_next_sparse_x_y {
                        let (next_range_image, round_poly) =
                            self.fuse_full_sparse_x_y_and_compute_round(&range_image, r);
                        self.cached_round_poly = Some(round_poly);
                        LowBasisRangeImageStorage::Materialized(next_range_image)
                    } else {
                        let next_range_image = Self::fold_range_image_sparse_x_y(
                            &range_image,
                            self.live_x_cols,
                            y_len,
                            r,
                        );
                        LowBasisRangeImageStorage::Materialized(next_range_image)
                    }
                } else {
                    let mut range_image = range_image;
                    fold_evals_in_place(&mut range_image, r);
                    LowBasisRangeImageStorage::Materialized(range_image)
                }
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

#[cfg(test)]
pub(crate) fn pad_compact_witness(
    digit_witness_prefix: &[i8],
    live_x_cols: usize,
    col_bits: usize,
    ring_bits: usize,
) -> Vec<i8> {
    let x_len = 1usize << col_bits;
    let y_len = 1usize << ring_bits;
    let mut padded = vec![0i8; x_len * y_len];
    for x in 0..live_x_cols {
        let offset = x * y_len;
        padded[offset..offset + y_len]
            .copy_from_slice(&digit_witness_prefix[offset..offset + y_len]);
    }
    padded
}
