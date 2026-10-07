use super::*;
use akita_error::checked;

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
        let basis = plan.basis();
        let range_poly = RangePoly::new(basis)?;
        let num_vars = col_bits.checked_add(ring_bits).ok_or_else(|| {
            AkitaError::InvalidInput("stage-1 challenge width overflow".to_string())
        })?;
        let x_len = checked::pow2(col_bits)
            .ok_or_else(|| AkitaError::InvalidInput("stage-1 column width overflow".to_string()))?;
        if live_x_cols == 0 || live_x_cols > x_len {
            return Err(AkitaError::InvalidSize {
                expected: x_len,
                actual: live_x_cols,
            });
        }
        let y_len = checked::pow2(ring_bits)
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
        // An octet prefix allocates a class table even for a tiny live witness,
        // so small witnesses stay on the field table. Basis 4 requires eight
        // live digits per class. The basis-8 threshold of four live digits per
        // class was chosen from one measured shape with 521,216 live digits,
        // where the class table was about 3.7x faster single-threaded and
        // slightly faster on 16 threads.
        let octet_minimum_digits = match basis {
            4 => 8 * 256,
            8 => 4 * 65_536,
            _ => usize::MAX,
        };
        let split_eq = GruenSplitEq::new(tau0)?;
        let range_image =
            if num_vars >= octet_prefix::OCTET_PREFIX_ROUNDS && expected >= octet_minimum_digits {
                LowBasisRangeImageStorage::OctetPrefix(OctetPrefix::new(
                    digit_witness,
                    tau0,
                    &split_eq,
                    basis,
                    range_poly,
                )?)
            } else {
                // Ring bits are low: retain only the flat live prefix, just as
                // octet-prefix materialization does. The omitted tail is zero.
                LowBasisRangeImageStorage::Materialized(
                    digit_witness
                        .iter()
                        .map(|digit| E::from_i64(i64::from(range_image_from_digit(digit))))
                        .collect(),
                )
            };
        Ok(Self {
            range_image,
            split_eq,
            range_poly,
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
    /// # Errors
    ///
    /// Returns an internal error if the virtual table has not been fully folded.
    pub fn final_range_image_eval(&self) -> Result<E, AkitaError> {
        match &self.range_image {
            LowBasisRangeImageStorage::Materialized(range_image) => match range_image.as_slice() {
                [value] => Ok(*value),
                _ => Err(AkitaError::Internal(format!(
                    "range-image final table length: expected 1, actual {}",
                    range_image.len(),
                ))),
            },
            LowBasisRangeImageStorage::OctetPrefix(_) => Err(AkitaError::Internal(
                "range image stayed in the octet prefix after the final fold".into(),
            )),
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
