use super::*;

impl<E: Field + Ring + Unreduced> LowBasisRangeCheckProver<E> {
    /// Build the low-basis prover from the compact witness table.
    ///
    /// `digit_witness` holds the `live_x_cols * 2^ring_bits` live digits in
    /// flat column-major layout, `tau0` is the stage-1 equality point in
    /// physical-address binding order (the coordinates of a
    /// [`DigitRangeEqualityPoint`](akita_types::DigitRangeEqualityPoint)
    /// over `col_bits + ring_bits` variables),
    /// and `plan` must be a direct-leaf plan (basis 4 or 8, no product
    /// stages).
    ///
    /// # Errors
    ///
    /// Returns an error if the plan has product stages, if any declared
    /// width overflows, or if the witness or `tau0` length disagrees with
    /// the declared `(live_x_cols, col_bits, ring_bits)` shape.
    pub(crate) fn new(
        digit_witness: PackedSignedDigits,
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
        let range_image = if num_vars >= octet_prefix::OCTET_PREFIX_ROUNDS {
            LowBasisRangeImageStorage::OctetPrefix(OctetPrefix {
                digits: digit_witness,
                tau: tau0.to_vec(),
                state: None,
            })
        } else {
            // Ring bits are low: retain only the flat live prefix, just as
            // octet-prefix materialization does. The omitted tail is zero.
            LowBasisRangeImageStorage::Materialized(
                (0..digit_witness.len())
                    .map(|index| {
                        E::from_i64(i64::from(range_image_from_digit(
                            digit_witness
                                .get(index)
                                .expect("validated live digit index"),
                        )))
                    })
                    .collect(),
            )
        };
        Ok(Self {
            range_image,
            split_eq: GruenSplitEq::new(tau0)?,
            range_poly: RangePoly::new(basis),
            live_x_cols,
            col_bits,
            num_vars,
            basis,
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
    pub fn final_range_image_eval(&self) -> E {
        match &self.range_image {
            LowBasisRangeImageStorage::Materialized(range_image) => {
                assert_eq!(range_image.len(), 1, "range_image not fully folded");
                range_image[0]
            }
            LowBasisRangeImageStorage::OctetPrefix(_) => {
                panic!("range_image stayed in the octet prefix after the final fold")
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

    /// Select a current or future round using the geometry after preceding binds.
    #[inline]
    pub(super) fn round_kernel(&self, round: usize) -> RoundKernel {
        debug_assert!(round >= self.rounds_completed);
        if round >= self.num_vars {
            return RoundKernel::Dense;
        }
        if round < octet_prefix::OCTET_PREFIX_ROUNDS
            && matches!(self.range_image, LowBasisRangeImageStorage::OctetPrefix(_))
        {
            return RoundKernel::OctetPrefix;
        }
        let bound_columns = round.saturating_sub(self.ring_bits());
        let current_bound_columns = self.rounds_completed.saturating_sub(self.ring_bits());
        let live_columns = self
            .live_x_cols
            .div_ceil(1usize << (bound_columns - current_bound_columns));
        let column_capacity = 1usize << (self.col_bits - bound_columns);
        if live_columns < column_capacity {
            RoundKernel::LivePrefix
        } else {
            RoundKernel::Dense
        }
    }
}
