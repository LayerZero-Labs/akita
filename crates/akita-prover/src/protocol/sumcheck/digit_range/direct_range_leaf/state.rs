use super::*;

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps> LowBasisRangeCheckProver<E> {
    /// Build the low-basis prover from the compact witness table.
    pub(crate) fn new(
        digit_witness: std::sync::Arc<[i8]>,
        tau0: &[E],
        plan: DigitRangePlan,
        live_x_cols: usize,
        col_bits: usize,
        ring_bits: usize,
    ) -> Result<Self, AkitaError> {
        if !plan.product_stage_arities().is_empty() {
            return Err(AkitaError::InvalidInput(
                "direct range prover requires basis 4 or 8".to_string(),
            ));
        }
        let basis = plan.basis();
        let num_vars = col_bits.checked_add(ring_bits).ok_or_else(|| {
            AkitaError::InvalidInput("stage-1 challenge width overflow".to_string())
        })?;
        let col_bits_u32 = u32::try_from(col_bits)
            .map_err(|_| AkitaError::InvalidInput("stage-1 column width overflow".to_string()))?;
        let x_len = 1usize
            .checked_shl(col_bits_u32)
            .ok_or_else(|| AkitaError::InvalidInput("stage-1 column width overflow".to_string()))?;
        if live_x_cols == 0 || live_x_cols > x_len {
            return Err(AkitaError::InvalidSize {
                expected: x_len,
                actual: live_x_cols,
            });
        }
        let ring_bits_u32 = u32::try_from(ring_bits)
            .map_err(|_| AkitaError::InvalidInput("stage-1 ring width overflow".to_string()))?;
        let y_len = 1usize
            .checked_shl(ring_bits_u32)
            .ok_or_else(|| AkitaError::InvalidInput("stage-1 ring width overflow".to_string()))?;
        let expected = live_x_cols
            .checked_mul(y_len)
            .ok_or_else(|| AkitaError::InvalidInput("stage-1 witness size overflow".to_string()))?;
        if digit_witness.len() != expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: digit_witness.len(),
            });
        }
        if tau0.len() != num_vars {
            return Err(AkitaError::InvalidSize {
                expected: num_vars,
                actual: tau0.len(),
            });
        }
        Ok(Self {
            range_image: LowBasisRangeImageStorage::Compact(digit_witness),
            split_eq: GruenSplitEq::new(tau0)?,
            polynomial_precomputation: RangePolynomialPrecomputation::new(basis),
            live_x_cols,
            col_bits,
            num_vars,
            basis,
            prefix_tau: can_use_stage1_two_round_prefix(ring_bits, basis).then(|| tau0.to_vec()),
            initial_round_prefix: None,
            cached_round_poly: None,
            rounds_completed: 0,
        })
    }

    /// Return `range_image(stage1_point)` after the final fold.
    ///
    /// # Panics
    ///
    /// Panics if called before the virtual table has been fully folded to a
    /// single field element.
    pub(crate) fn final_range_image_eval(&self) -> E {
        match &self.range_image {
            LowBasisRangeImageStorage::Materialized(range_image) => {
                assert_eq!(range_image.len(), 1, "range_image not fully folded");
                range_image[0]
            }
            LowBasisRangeImageStorage::Compact(_) => {
                panic!("range_image remained compact after final fold")
            }
        }
    }

    #[inline]
    pub(super) fn ring_bits(&self) -> usize {
        self.num_vars - self.col_bits
    }

    #[inline]
    pub(super) fn in_x_phase(&self) -> bool {
        self.rounds_completed >= self.ring_bits()
    }

    #[inline]
    pub(super) fn current_x_width(&self) -> usize {
        debug_assert!(self.in_x_phase());
        self.num_vars.saturating_sub(self.rounds_completed)
    }

    #[inline]
    pub(super) fn current_x_len(&self) -> usize {
        1usize << self.current_x_width()
    }

    #[inline]
    pub(super) fn use_prefix_x_round(&self) -> bool {
        self.in_x_phase() && self.live_x_cols < self.current_x_len()
    }

    #[inline]
    pub(super) fn next_use_prefix_x_round_after_current(&self) -> bool {
        self.in_x_phase()
            && self.rounds_completed + 1 < self.num_vars
            && self.live_x_cols.div_ceil(2) < (self.current_x_len() / 2)
    }

    #[inline]
    pub(super) fn next_use_sparse_x_y_round_after_current(&self) -> bool {
        !self.in_x_phase() && self.rounds_completed + 1 < self.ring_bits()
    }

    #[inline]
    pub(crate) fn can_use_two_round_prefix(&self) -> bool {
        self.prefix_tau.is_some()
    }

    #[inline]
    pub(super) fn using_two_round_prefix(&self) -> bool {
        self.rounds_completed < 2 && self.can_use_two_round_prefix()
    }

    #[inline]
    pub(super) fn defers_binary_range_image_through_third_round(&self) -> bool {
        self.basis == 4 && self.ring_bits() >= 3 && self.can_use_two_round_prefix()
    }

    #[inline]
    pub(super) fn awaiting_binary_range_image_third_challenge(&self) -> bool {
        self.rounds_completed == 2
            && self.defers_binary_range_image_through_third_round()
            && matches!(self.range_image, LowBasisRangeImageStorage::Compact(_))
    }

    #[inline]
    pub(super) fn valid_range_image_values(basis: usize) -> Vec<i16> {
        let half = (basis / 2) as i16;
        (0..half).map(|k| k * (k + 1)).collect()
    }

    #[inline]
    pub(super) fn build_range_image_fold_lut(basis: usize, r: E) -> CompactPairFoldLut<E> {
        let valid_range_images = Self::valid_range_image_values(basis);
        CompactPairFoldLut::from_allowed_values(&valid_range_images, r)
    }

    pub(super) fn ensure_initial_round_prefix(&mut self) -> &mut DirectRangePrefixState<E> {
        if self.initial_round_prefix.is_none() {
            let tau0 = self
                .prefix_tau
                .clone()
                .expect("two-round prefix requested without cached tau");
            let ring_bits = self.num_vars - self.col_bits;
            let compact_range_image = match &self.range_image {
                LowBasisRangeImageStorage::Compact(digit_witness) => digit_witness.as_ref(),
                LowBasisRangeImageStorage::Materialized(_) => {
                    panic!("two-round prefix can only build from compact table")
                }
            };
            let proof = build_stage1_bivariate_skip_proof_from_compact_range_image(
                compact_range_image,
                &tau0,
                self.basis,
                self.live_x_cols,
                self.col_bits,
                ring_bits,
            )
            .expect("two-round prefix should be available");
            let skip_state = Stage1BivariateSkipState::new(&proof, &tau0, self.basis)
                .expect("valid bivariate-skip state");
            self.initial_round_prefix = Some(DirectRangePrefixState {
                skip_state,
                first_challenge: None,
                second_challenge: None,
            });
        }
        self.initial_round_prefix
            .as_mut()
            .expect("two-round prefix should be initialized")
    }
}
