use super::*;

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps> DirectRangeLeafState<E> {
    pub(super) fn compute_current_round_eq_poly_from_state(&mut self) -> EqFactoredUniPoly<E> {
        let use_two_round_prefix = self.using_two_round_prefix();
        let use_prefix_x_round = !use_two_round_prefix && self.use_prefix_x_round();
        let use_sparse_x_y_round = !use_two_round_prefix && self.use_sparse_x_y_round();
        let t_round = Instant::now();
        let rounds_completed = self.rounds_completed;
        let poly = if use_two_round_prefix {
            let prefix = self.ensure_two_round_prefix();
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
                RangeImageState::Compact => {
                    let compact_range_image = &self.digit_witness;
                    if use_prefix_x_round {
                        self.compute_round_compact_prefix_x(compact_range_image)
                    } else if use_sparse_x_y_round {
                        self.compute_round_compact_sparse_x_y(compact_range_image)
                    } else {
                        compute_norm_round_eq_poly_from_compact_range_image(
                            &self.split_eq,
                            compact_range_image,
                            &self.range_precomp,
                        )
                    }
                }
                RangeImageState::Full(range_image) => {
                    if use_prefix_x_round {
                        self.compute_round_full_prefix_x(range_image)
                    } else if use_sparse_x_y_round {
                        self.compute_round_full_sparse_x_y(range_image)
                    } else {
                        compute_norm_round_eq_poly_from_s(
                            &self.split_eq,
                            &self.range_precomp,
                            |j| (range_image[2 * j], range_image[2 * j + 1]),
                        )
                    }
                }
            }
        };

        if use_two_round_prefix || use_prefix_x_round {
            self.prefix_time_total += t_round.elapsed().as_secs_f64();
        } else {
            self.dense_time_total += t_round.elapsed().as_secs_f64();
        }

        poly
    }

    #[tracing::instrument(
        skip_all,
        name = "DirectRangeLeafState::fold_compact_range_image_to_full"
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
    EqFactoredSumcheckInstanceProver<E> for DirectRangeLeafState<E>
{
    fn num_rounds(&self) -> usize {
        self.num_vars
    }

    fn degree_bound(&self) -> usize {
        self.b / 2
    }

    fn input_claim(&self) -> E {
        E::zero()
    }

    fn current_linear_factor_evals(&self) -> (E, E) {
        self.split_eq.linear_factor_evals()
    }

    fn compute_round_eq_factored(&mut self, _round: usize) -> EqFactoredUniPoly<E> {
        if let Some(poly) = self.cached_round_poly.take() {
            poly
        } else {
            self.compute_current_round_eq_poly_from_state()
        }
    }

    fn ingest_challenge(&mut self, _round: usize, r: E) {
        let t_fold = Instant::now();
        let _span = tracing::info_span!("DirectRangeLeafState::fold_round").entered();
        if self.using_two_round_prefix() {
            let rounds_completed = self.rounds_completed;
            self.split_eq.bind(r);
            if rounds_completed == 0 {
                self.ensure_two_round_prefix().first_challenge = Some(r);
            } else {
                let r0 = {
                    let prefix = self.ensure_two_round_prefix();
                    prefix
                        .first_challenge
                        .expect("round 1 ingest requires the round 0 challenge")
                };
                let y_len = match &self.range_image {
                    RangeImageState::Compact => self.digit_witness.len() / self.live_x_cols,
                    RangeImageState::Full(_) => panic!("two-round prefix expected compact table"),
                };
                self.range_image = match std::mem::replace(
                    &mut self.range_image,
                    RangeImageState::Full(Vec::new()),
                ) {
                    RangeImageState::Compact => {
                        let compact_range_image = &self.digit_witness;
                        if self.ring_bits() > 2 {
                            let (range_image, round_poly) = self
                                .fuse_compact_to_round2_and_compute_round(
                                    compact_range_image,
                                    r0,
                                    r,
                                );
                            self.cached_round_poly = Some(round_poly);
                            RangeImageState::Full(range_image)
                        } else {
                            let range_image = Self::fold_compact_range_image_to_round2(
                                compact_range_image,
                                self.live_x_cols,
                                y_len,
                                r0,
                                r,
                            );
                            RangeImageState::Full(range_image)
                        }
                    }
                    RangeImageState::Full(_) => {
                        unreachable!("two-round prefix should hold compact table")
                    }
                };
            }
            self.rounds_completed += 1;
            if self.rounds_completed < self.num_vars {
                if self.cached_round_poly.is_none() {
                    self.cached_round_poly = Some(self.compute_current_round_eq_poly_from_state());
                }
            } else {
                self.cached_round_poly = None;
            }
            drop(_span);
            self.fold_time_total += t_fold.elapsed().as_secs_f64();
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
            RangeImageState::Compact => self.digit_witness.len() / self.live_x_cols,
            RangeImageState::Full(range_image) => range_image.len() / self.live_x_cols,
        };

        self.range_image =
            match std::mem::replace(&mut self.range_image, RangeImageState::Full(Vec::new())) {
                RangeImageState::Compact => {
                    let compact_range_image = &self.digit_witness;
                    let fold_lut = Self::build_range_image_fold_lut(self.b, r);
                    let range_image = if use_prefix_x_round {
                        Self::fold_compact_range_image_prefix_x(
                            compact_range_image,
                            self.live_x_cols,
                            y_len,
                            &fold_lut,
                        )
                    } else {
                        Self::fold_compact_range_image_to_full(compact_range_image, &fold_lut)
                    };
                    RangeImageState::Full(range_image)
                }
                RangeImageState::Full(range_image) => {
                    if use_prefix_x_round {
                        if fuse_next_full_prefix_x {
                            let (next_range_image, round_poly) =
                                self.fuse_full_prefix_x_and_compute_round(&range_image, r);
                            self.cached_round_poly = Some(round_poly);
                            RangeImageState::Full(next_range_image)
                        } else {
                            let next_range_image = Self::fold_range_image_prefix_x(
                                &range_image,
                                self.live_x_cols,
                                y_len,
                                r,
                            );
                            RangeImageState::Full(next_range_image)
                        }
                    } else if use_sparse_x_y_round {
                        if fuse_next_sparse_x_y {
                            let (next_range_image, round_poly) =
                                self.fuse_full_sparse_x_y_and_compute_round(&range_image, r);
                            self.cached_round_poly = Some(round_poly);
                            RangeImageState::Full(next_range_image)
                        } else {
                            let next_range_image = Self::fold_range_image_sparse_x_y(
                                &range_image,
                                self.live_x_cols,
                                y_len,
                                r,
                            );
                            RangeImageState::Full(next_range_image)
                        }
                    } else {
                        let mut range_image = range_image;
                        fold_evals_in_place(&mut range_image, r);
                        RangeImageState::Full(range_image)
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
        drop(_span);
        self.fold_time_total += t_fold.elapsed().as_secs_f64();
    }

    fn finalize(&mut self) {
        tracing::debug!(
            rounds = self.num_vars,
            prefix_s = self.prefix_time_total,
            dense_s = self.dense_time_total,
            fold_range_image = self.fold_time_total,
            "stage1 sumcheck rounds complete"
        );
    }
}

#[cfg(test)]
pub(crate) fn pad_compact_witness(
    w_prefix: &[i8],
    live_x_cols: usize,
    col_bits: usize,
    ring_bits: usize,
) -> Vec<i8> {
    let x_len = 1usize << col_bits;
    let y_len = 1usize << ring_bits;
    let mut padded = vec![0i8; x_len * y_len];
    for x in 0..live_x_cols {
        let offset = x * y_len;
        padded[offset..offset + y_len].copy_from_slice(&w_prefix[offset..offset + y_len]);
    }
    padded
}

#[cfg(test)]
pub(crate) fn advance_stage1_claim<
    F: FieldCore + FromPrimitiveInt + akita_field::CanonicalField + HasUnreducedOps + HasOptimizedFold,
>(
    prover: &DirectRangeLeafState<F>,
    scaled_claim: F,
    claim_scale: F,
    poly: &EqFactoredUniPoly<F>,
    challenge: F,
) -> (F, F) {
    use akita_sumcheck::advance_eq_factored_claim;
    let (l_at_0, l_at_1) = prover.current_linear_factor_evals();
    advance_eq_factored_claim(scaled_claim, claim_scale, l_at_0, l_at_1, poly, challenge)
}
