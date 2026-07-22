use super::utils::{accumulate_left_round_eq, fold_dense_left_round};
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::poly::multilinear_eval;
use akita_algebra::uni_poly::UniPoly;
use akita_field::parallel::*;
use akita_field::unreduced::HasUnreducedOps;
use akita_field::{AkitaError, FieldCore, FromPrimitiveInt, Zero};
use akita_sumcheck::reduce_signed_accum;
use std::sync::Arc;

// Selected independently by `stage3_small_digit_kernel_ab`. These are physical
// implementation tiles, not protocol geometry. LB2 benefits from four-way
// matching and small register tiles in both layouts. The generic kernel keeps
// the existing coarser task tile rather than adding a density-based selector.
const LB2_COLUMN_OUTPUT_TILE: usize = 4;
const LB2_ROW_OUTPUT_TILE: usize = 4;
const DIRECT_COLUMN_OUTPUT_TILE: usize = 32;
const DIRECT_ROW_OUTPUT_TILE: usize = 32;

/// Two-pass prefix/suffix prover for one linear witness-opening reduction.
///
/// The source stays in compact signed-digit form. Phase 1 contracts the suffix
/// of the old opening point in one source pass and proves the prefix rounds from
/// square-root state. Phase 2 makes one more source pass after the prefix
/// challenges are known, then proves the suffix rounds from the folded witness.
pub(super) struct WitnessClaimReductionTerm<E: FieldCore> {
    source: Arc<[i8]>,
    phase: WitnessClaimReductionPhase<E>,
    input_claim: E,
    prefix_rounds: usize,
    total_rounds: usize,
    log_basis: u32,
    max_abs_digit: u8,
}

enum WitnessClaimReductionPhase<E: FieldCore> {
    Prefix {
        table: Vec<E>,
        equality_point: Arc<[E]>,
        equality_scale: E,
        suffix_point: Arc<[E]>,
        challenges: Vec<E>,
    },
    Suffix {
        table: Vec<E>,
        equality_point: Arc<[E]>,
        equality_scale: E,
    },
}

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps> WitnessClaimReductionTerm<E> {
    pub(super) fn new(
        source: Arc<[i8]>,
        padded_len: usize,
        old_point: Arc<[E]>,
        log_basis: u32,
        observed_max_abs_digit: u8,
    ) -> Result<Self, AkitaError> {
        if padded_len < 2 || !padded_len.is_power_of_two() || source.len() != padded_len {
            return Err(AkitaError::InvalidInput(
                "witness claim reduction requires a complete power-of-two compact domain".into(),
            ));
        }
        let total_rounds = padded_len.trailing_zeros() as usize;
        if old_point.len() != total_rounds {
            return Err(AkitaError::InvalidPointDimension {
                expected: total_rounds,
                actual: old_point.len(),
            });
        }
        let (min_digit, max_digit) = balanced_digit_bounds(log_basis)?;
        let certified_max_abs_digit = min_digit.unsigned_abs();
        if observed_max_abs_digit > certified_max_abs_digit {
            return Err(AkitaError::InvalidInput(format!(
                "stage-3 witness digit exceeds balanced log-basis-{log_basis} range"
            )));
        }
        debug_assert!(
            source.iter().all(|digit| {
                (min_digit..=max_digit).contains(digit)
                    && digit.unsigned_abs() <= observed_max_abs_digit
            }),
            "the caller must validate the exact balanced range while remapping the witness"
        );
        let max_abs_digit = observed_max_abs_digit;
        let prefix_rounds = total_rounds / 2;
        let prefix_len = 1usize << prefix_rounds;
        let suffix_point: Arc<[E]> = Arc::from(old_point[prefix_rounds..].to_vec());
        let suffix_evals = EqPolynomial::evals(&suffix_point)?;
        validate_unreduced_headroom(suffix_evals.len(), max_abs_digit)?;
        let table = {
            let kernel = if log_basis == 2 {
                "lb2_match_unreduced"
            } else {
                "direct_small_unreduced"
            };
            let _span = tracing::info_span!(
                "stage3_witness_prefix_pass",
                kernel,
                log_basis,
                certified_max_abs_digit,
                observed_max_abs_digit
            )
            .entered();
            if log_basis == 2 {
                contract_columns_lb2::<E, LB2_COLUMN_OUTPUT_TILE>(
                    &source,
                    &suffix_evals,
                    prefix_len,
                )
            } else {
                contract_columns_small::<E, DIRECT_COLUMN_OUTPUT_TILE>(
                    &source,
                    &suffix_evals,
                    prefix_len,
                )
            }
        };
        let input_claim = multilinear_eval(&table, &old_point[..prefix_rounds])?;
        let phase = if prefix_rounds == 0 {
            Self::build_suffix_phase(
                &source,
                &[],
                suffix_point,
                E::one(),
                log_basis,
                max_abs_digit,
            )?
        } else {
            WitnessClaimReductionPhase::Prefix {
                table,
                equality_point: Arc::from(old_point[..prefix_rounds].to_vec()),
                equality_scale: E::one(),
                suffix_point,
                challenges: Vec::with_capacity(prefix_rounds),
            }
        };
        Ok(Self {
            source,
            phase,
            input_claim,
            prefix_rounds,
            total_rounds,
            log_basis,
            max_abs_digit,
        })
    }

    pub(super) const fn num_rounds(&self) -> usize {
        self.total_rounds
    }

    pub(super) const fn input_claim(&self) -> E {
        self.input_claim
    }

    pub(super) fn compute_round_univariate(&self) -> UniPoly<E> {
        let (table, equality_point, equality_scale) = match &self.phase {
            WitnessClaimReductionPhase::Prefix {
                table,
                equality_point,
                equality_scale,
                ..
            }
            | WitnessClaimReductionPhase::Suffix {
                table,
                equality_point,
                equality_scale,
            } => (table, equality_point, *equality_scale),
        };
        let (constant, linear, quadratic) =
            accumulate_left_round_eq(table, equality_point, equality_scale, E::one());
        UniPoly::from_coeffs(vec![constant, linear, quadratic])
    }

    pub(super) fn ingest_challenge(
        &mut self,
        round: usize,
        challenge: E,
    ) -> Result<(), AkitaError> {
        match &mut self.phase {
            WitnessClaimReductionPhase::Prefix {
                table,
                equality_point,
                equality_scale,
                suffix_point,
                challenges,
            } => {
                if round >= self.prefix_rounds {
                    return Err(AkitaError::InvalidProof);
                }
                fold_dense_left_round(table, challenge);
                fold_equality_factor(equality_point, equality_scale, challenge)?;
                challenges.push(challenge);
                if challenges.len() == self.prefix_rounds {
                    let prefix_scale = *equality_scale;
                    let suffix_point = Arc::clone(suffix_point);
                    let prefix_challenges = std::mem::take(challenges);
                    self.phase = Self::build_suffix_phase(
                        &self.source,
                        &prefix_challenges,
                        suffix_point,
                        prefix_scale,
                        self.log_basis,
                        self.max_abs_digit,
                    )?;
                }
            }
            WitnessClaimReductionPhase::Suffix {
                table,
                equality_point,
                equality_scale,
            } => {
                if round < self.prefix_rounds || round >= self.total_rounds {
                    return Err(AkitaError::InvalidProof);
                }
                fold_dense_left_round(table, challenge);
                fold_equality_factor(equality_point, equality_scale, challenge)?;
            }
        }
        Ok(())
    }

    pub(super) fn folded_witness_value(&self) -> Result<E, AkitaError> {
        let table = match &self.phase {
            WitnessClaimReductionPhase::Prefix { table, .. }
            | WitnessClaimReductionPhase::Suffix { table, .. } => table,
        };
        if table.len() != 1 {
            return Err(AkitaError::InvalidSize {
                expected: 1,
                actual: table.len(),
            });
        }
        Ok(table[0])
    }

    fn build_suffix_phase(
        source: &[i8],
        prefix_challenges: &[E],
        suffix_point: Arc<[E]>,
        prefix_scale: E,
        log_basis: u32,
        max_abs_digit: u8,
    ) -> Result<WitnessClaimReductionPhase<E>, AkitaError> {
        let prefix_evals = EqPolynomial::evals(prefix_challenges)?;
        let prefix_len = prefix_evals.len();
        if !source.len().is_multiple_of(prefix_len) {
            return Err(AkitaError::InvalidProof);
        }
        let suffix_len = source.len() / prefix_len;
        validate_unreduced_headroom(prefix_len, max_abs_digit)?;
        let table = {
            let kernel = if log_basis == 2 {
                "lb2_match_unreduced"
            } else {
                "direct_small_unreduced"
            };
            let _span = tracing::info_span!(
                "stage3_witness_suffix_pass",
                kernel,
                log_basis,
                certified_max_abs_digit = balanced_digit_abs_bound(log_basis)?,
                observed_max_abs_digit = max_abs_digit
            )
            .entered();
            if log_basis == 2 {
                contract_rows_lb2::<E, LB2_ROW_OUTPUT_TILE>(source, &prefix_evals, suffix_len)
            } else {
                contract_rows_small::<E, DIRECT_ROW_OUTPUT_TILE>(source, &prefix_evals, suffix_len)
            }
        };
        Ok(WitnessClaimReductionPhase::Suffix {
            table,
            equality_point: suffix_point,
            equality_scale: prefix_scale,
        })
    }
}

#[inline(always)]
fn accumulate_signed_small<E: FieldCore + HasUnreducedOps>(
    positive: &mut E::MulU64Accum,
    negative: &mut E::MulU64Accum,
    weight: E,
    digit: i8,
) {
    let magnitude = u64::from(digit.unsigned_abs());
    if magnitude == 0 {
        return;
    }
    let product = weight.mul_u64_unreduced(magnitude);
    if digit.is_negative() {
        *negative += product;
    } else {
        *positive += product;
    }
}

#[inline(always)]
fn accumulate_lb2_unit<A: Copy + std::ops::AddAssign>(
    positive: &mut A,
    negative: &mut A,
    unit: A,
    digit: i8,
) {
    match digit {
        -2 => {
            *negative += unit;
            *negative += unit;
        }
        -1 => *negative += unit,
        0 => {}
        1 => *positive += unit,
        _ => unreachable!("log-basis-2 digits are validated before contraction"),
    }
}

fn contract_columns_small<E, const TILE: usize>(
    source: &[i8],
    weights: &[E],
    output_len: usize,
) -> Vec<E>
where
    E: FieldCore + HasUnreducedOps,
{
    let mut output = vec![E::zero(); output_len];
    cfg_chunks_mut!(&mut output, TILE)
        .enumerate()
        .for_each(|(chunk_index, chunk)| {
            let output_start = chunk_index * TILE;
            let mut positive = [E::MulU64Accum::zero(); TILE];
            let mut negative = [E::MulU64Accum::zero(); TILE];
            for (row, &weight) in weights.iter().enumerate() {
                let source_start = row * output_len + output_start;
                for offset in 0..chunk.len() {
                    accumulate_signed_small::<E>(
                        &mut positive[offset],
                        &mut negative[offset],
                        weight,
                        source[source_start + offset],
                    );
                }
            }
            for (offset, slot) in chunk.iter_mut().enumerate() {
                *slot = reduce_signed_accum::<E>(positive[offset], negative[offset]);
            }
        });
    output
}

fn contract_columns_lb2<E, const TILE: usize>(
    source: &[i8],
    weights: &[E],
    output_len: usize,
) -> Vec<E>
where
    E: FieldCore + HasUnreducedOps,
{
    let mut output = vec![E::zero(); output_len];
    cfg_chunks_mut!(&mut output, TILE)
        .enumerate()
        .for_each(|(chunk_index, chunk)| {
            let output_start = chunk_index * TILE;
            let mut positive = [E::MulU64Accum::zero(); TILE];
            let mut negative = [E::MulU64Accum::zero(); TILE];
            for (row, &weight) in weights.iter().enumerate() {
                let source_start = row * output_len + output_start;
                let unit = weight.mul_u64_unreduced(1);
                for offset in 0..chunk.len() {
                    accumulate_lb2_unit(
                        &mut positive[offset],
                        &mut negative[offset],
                        unit,
                        source[source_start + offset],
                    );
                }
            }
            for (offset, slot) in chunk.iter_mut().enumerate() {
                *slot = reduce_signed_accum::<E>(positive[offset], negative[offset]);
            }
        });
    output
}

fn contract_rows_small<E, const TILE: usize>(
    source: &[i8],
    weights: &[E],
    output_len: usize,
) -> Vec<E>
where
    E: FieldCore + HasUnreducedOps,
{
    let mut output = vec![E::zero(); output_len];
    cfg_chunks_mut!(&mut output, TILE)
        .enumerate()
        .for_each(|(chunk_index, chunk)| {
            let output_start = chunk_index * TILE;
            for (offset, slot) in chunk.iter_mut().enumerate() {
                let source_start = (output_start + offset) * weights.len();
                let mut positive = E::MulU64Accum::zero();
                let mut negative = E::MulU64Accum::zero();
                for (column, &weight) in weights.iter().enumerate() {
                    accumulate_signed_small::<E>(
                        &mut positive,
                        &mut negative,
                        weight,
                        source[source_start + column],
                    );
                }
                *slot = reduce_signed_accum::<E>(positive, negative);
            }
        });
    output
}

fn contract_rows_lb2<E, const TILE: usize>(
    source: &[i8],
    weights: &[E],
    output_len: usize,
) -> Vec<E>
where
    E: FieldCore + HasUnreducedOps,
{
    let mut output = vec![E::zero(); output_len];
    cfg_chunks_mut!(&mut output, TILE)
        .enumerate()
        .for_each(|(chunk_index, chunk)| {
            let output_start = chunk_index * TILE;
            let mut positive = [E::MulU64Accum::zero(); TILE];
            let mut negative = [E::MulU64Accum::zero(); TILE];
            for (column, &weight) in weights.iter().enumerate() {
                let unit = weight.mul_u64_unreduced(1);
                for offset in 0..chunk.len() {
                    let source_index = (output_start + offset) * weights.len() + column;
                    accumulate_lb2_unit(
                        &mut positive[offset],
                        &mut negative[offset],
                        unit,
                        source[source_index],
                    );
                }
            }
            for (offset, slot) in chunk.iter_mut().enumerate() {
                *slot = reduce_signed_accum::<E>(positive[offset], negative[offset]);
            }
        });
    output
}

pub(super) fn balanced_digit_bounds(log_basis: u32) -> Result<(i8, i8), AkitaError> {
    if !(2..=6).contains(&log_basis) {
        return Err(AkitaError::InvalidSetup(format!(
            "stage-3 witness claim reduction requires protocol log basis in 2..=6, got {log_basis}"
        )));
    }
    let half = 1i16 << (log_basis - 1);
    Ok(((-half) as i8, (half - 1) as i8))
}

pub(super) fn balanced_digit_abs_bound(log_basis: u32) -> Result<u8, AkitaError> {
    let (min, _) = balanced_digit_bounds(log_basis)?;
    Ok(min.unsigned_abs())
}

fn validate_unreduced_headroom(
    contraction_len: usize,
    max_abs_digit: u8,
) -> Result<(), AkitaError> {
    let additions = (contraction_len as u128)
        .checked_mul(u128::from(max_abs_digit))
        .ok_or_else(|| AkitaError::InvalidSetup("stage-3 contraction size overflow".into()))?;
    if additions > u128::from(u64::MAX) {
        return Err(AkitaError::InvalidSetup(
            "stage-3 witness contraction exceeds unreduced accumulator headroom".into(),
        ));
    }
    Ok(())
}

fn fold_equality_factor<E: FieldCore>(
    point: &mut Arc<[E]>,
    scale: &mut E,
    challenge: E,
) -> Result<(), AkitaError> {
    let (&head, tail) = point.split_first().ok_or(AkitaError::InvalidProof)?;
    *scale *= (E::one() - challenge) * (E::one() - head) + challenge * head;
    *point = Arc::from(tail.to_vec());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::sumcheck::akita_stage3::utils::{
        accumulate_left_round, fold_factor_in_place, product_claim,
    };
    use akita_field::Prime128Offset275;
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    type F = Prime128Offset275;

    fn test_point(rounds: usize) -> Vec<F> {
        (0..rounds)
            .map(|round| F::from_u64(3 + 2 * round as u64))
            .collect()
    }

    fn test_digits(len: usize) -> Vec<i8> {
        (0..len)
            .map(|index| match index % 7 {
                0 | 5 => 0,
                1 => 1,
                2 => -1,
                3 => 2,
                4 => -2,
                _ => 3,
            })
            .collect()
    }

    fn assert_matches_dense(digits: Vec<i8>, log_basis: u32) {
        let len = digits.len();
        let rounds = len.trailing_zeros() as usize;
        let point = test_point(rounds);
        let mut term = WitnessClaimReductionTerm::new(
            Arc::from(digits.clone()),
            len,
            Arc::from(point.clone()),
            log_basis,
            digits
                .iter()
                .map(|digit| digit.unsigned_abs())
                .max()
                .unwrap(),
        )
        .expect("valid prefix/suffix witness term");
        let mut dense_witness = digits
            .iter()
            .map(|&digit| F::from_i64(i64::from(digit)))
            .collect::<Vec<_>>();
        let mut dense_equality = EqPolynomial::evals(&point).expect("valid equality point");
        assert_eq!(
            term.input_claim(),
            product_claim(&dense_witness, &dense_equality, &[F::one()])
        );

        for round in 0..rounds {
            let got = term.compute_round_univariate();
            let expected = accumulate_left_round(&dense_witness, &dense_equality, F::one());
            assert_eq!(got.coeffs, vec![expected.0, expected.1, expected.2]);

            let challenge = F::from_u64(17 + round as u64);
            term.ingest_challenge(round, challenge)
                .expect("valid round challenge");
            fold_dense_left_round(&mut dense_witness, challenge);
            fold_factor_in_place(&mut dense_equality, challenge);
        }
        assert_eq!(
            term.folded_witness_value().expect("folded witness"),
            dense_witness[0]
        );
    }

    #[test]
    fn prefix_suffix_matches_dense_for_even_and_uneven_splits() {
        for len in [2, 4, 8, 16, 32, 64] {
            assert_matches_dense(test_digits(len), 3);
        }
    }

    #[test]
    fn lb2_match_kernel_covers_exact_balanced_digit_range() {
        for len in [2, 4, 8, 16, 32, 64] {
            let digits = (0..len)
                .map(|index| [-2, -1, 0, 1][index % 4])
                .collect::<Vec<_>>();
            assert_matches_dense(digits, 2);
        }
    }

    #[test]
    fn balanced_digit_bounds_are_asymmetric() {
        assert_eq!(balanced_digit_bounds(2).unwrap(), (-2, 1));
        assert_eq!(balanced_digit_bounds(3).unwrap(), (-4, 3));
        assert_eq!(balanced_digit_bounds(6).unwrap(), (-32, 31));
        for unsupported in [0, 1, 7, 8] {
            assert!(balanced_digit_bounds(unsupported).is_err());
        }
    }

    #[test]
    fn prefix_suffix_preserves_explicit_zero_padding() {
        let mut digits = test_digits(32);
        digits[19..].fill(0);
        let point = test_point(5);
        let term = WitnessClaimReductionTerm::new(
            Arc::from(digits.clone()),
            digits.len(),
            Arc::from(point.clone()),
            3,
            digits
                .iter()
                .map(|digit| digit.unsigned_abs())
                .max()
                .unwrap(),
        )
        .expect("valid padded witness term");
        let dense = digits
            .iter()
            .map(|&digit| F::from_i64(i64::from(digit)))
            .collect::<Vec<_>>();
        let expected = multilinear_eval(&dense, &point).expect("dense padded opening");
        assert_eq!(term.input_claim(), expected);
    }

    #[test]
    fn prefix_suffix_rejects_malformed_domains() {
        assert!(WitnessClaimReductionTerm::<F>::new(
            Arc::from(vec![1i8; 3]),
            3,
            Arc::from(test_point(2)),
            3,
            1,
        )
        .is_err());
        assert!(WitnessClaimReductionTerm::<F>::new(
            Arc::from(vec![1i8; 4]),
            8,
            Arc::from(test_point(3)),
            3,
            1,
        )
        .is_err());
        assert!(WitnessClaimReductionTerm::<F>::new(
            Arc::from(vec![1i8; 8]),
            8,
            Arc::from(test_point(2)),
            3,
            1,
        )
        .is_err());
    }

    fn median_duration(mut samples: Vec<Duration>) -> Duration {
        samples.sort_unstable();
        samples[samples.len() / 2]
    }

    fn median_run(mut run: impl FnMut(), samples: usize) -> Duration {
        let mut durations = Vec::with_capacity(samples);
        for _ in 0..samples {
            let start = Instant::now();
            run();
            durations.push(start.elapsed());
        }
        median_duration(durations)
    }

    fn report_lb2_tile<const TILE: usize>(
        shape: &str,
        distribution: &str,
        source: &[i8],
        weights: &[F],
        output_len: usize,
        direct: Duration,
        samples: usize,
    ) {
        let expected = if shape == "columns" {
            contract_columns_small::<F, DIRECT_COLUMN_OUTPUT_TILE>(source, weights, output_len)
        } else {
            contract_rows_small::<F, DIRECT_ROW_OUTPUT_TILE>(source, weights, output_len)
        };
        let actual = if shape == "columns" {
            contract_columns_lb2::<F, TILE>(source, weights, output_len)
        } else {
            contract_rows_lb2::<F, TILE>(source, weights, output_len)
        };
        assert_eq!(expected, actual);
        let lb2 = median_run(
            || {
                if shape == "columns" {
                    black_box(contract_columns_lb2::<F, TILE>(
                        black_box(source),
                        black_box(weights),
                        output_len,
                    ));
                } else {
                    black_box(contract_rows_lb2::<F, TILE>(
                        black_box(source),
                        black_box(weights),
                        output_len,
                    ));
                }
            },
            samples,
        );
        eprintln!(
            "{shape}\t{distribution}\tlb2\t{TILE}\t{:.3}\t{:.3}",
            lb2.as_secs_f64() * 1e3,
            lb2.as_secs_f64() / direct.as_secs_f64()
        );
    }

    fn report_direct_tile<const TILE: usize>(
        shape: &str,
        distribution: &str,
        source: &[i8],
        weights: &[F],
        output_len: usize,
        samples: usize,
    ) -> Duration {
        let elapsed = median_run(
            || {
                if shape == "columns" {
                    black_box(contract_columns_small::<F, TILE>(
                        black_box(source),
                        black_box(weights),
                        output_len,
                    ));
                } else {
                    black_box(contract_rows_small::<F, TILE>(
                        black_box(source),
                        black_box(weights),
                        output_len,
                    ));
                }
            },
            samples,
        );
        eprintln!(
            "{shape}\t{distribution}\tdirect\t{TILE}\t{:.3}\t-",
            elapsed.as_secs_f64() * 1e3
        );
        elapsed
    }

    /// Explicit A/B microbenchmark for generic and LB2-specific contractions.
    ///
    /// Run with:
    /// `cargo test -p akita-prover --release stage3_small_digit_kernel_ab -- --ignored --nocapture`
    #[test]
    #[ignore = "explicit performance experiment"]
    fn stage3_small_digit_kernel_ab() {
        const COLUMN_WEIGHT_COUNT: usize = 4096;
        const COLUMN_OUTPUT_COUNT: usize = 256;
        const ROW_WEIGHT_COUNT: usize = 1024;
        const ROW_OUTPUT_COUNT: usize = 1024;
        const SAMPLES: usize = 7;
        let weights = (0..COLUMN_WEIGHT_COUNT)
            .map(|index| F::from_u64(17 + index as u64 * 2))
            .collect::<Vec<_>>();
        let row_weights = &weights[..ROW_WEIGHT_COUNT];
        eprintln!("shape\tdistribution\tkernel\ttile\ttime_ms\tlb2/direct32");
        for distribution in ["dense", "sparse"] {
            let digit = |index: usize| {
                if distribution == "sparse" && index % 8 < 6 {
                    0
                } else {
                    [-2, -1, 0, 1][index % 4]
                }
            };
            let column_source = (0..COLUMN_WEIGHT_COUNT * COLUMN_OUTPUT_COUNT)
                .map(digit)
                .collect::<Vec<_>>();
            let row_source = (0..ROW_OUTPUT_COUNT * ROW_WEIGHT_COUNT)
                .map(digit)
                .collect::<Vec<_>>();
            report_direct_tile::<4>(
                "columns",
                distribution,
                &column_source,
                &weights,
                COLUMN_OUTPUT_COUNT,
                SAMPLES,
            );
            report_direct_tile::<8>(
                "columns",
                distribution,
                &column_source,
                &weights,
                COLUMN_OUTPUT_COUNT,
                SAMPLES,
            );
            report_direct_tile::<16>(
                "columns",
                distribution,
                &column_source,
                &weights,
                COLUMN_OUTPUT_COUNT,
                SAMPLES,
            );
            let column_direct = report_direct_tile::<32>(
                "columns",
                distribution,
                &column_source,
                &weights,
                COLUMN_OUTPUT_COUNT,
                SAMPLES,
            );
            report_direct_tile::<64>(
                "columns",
                distribution,
                &column_source,
                &weights,
                COLUMN_OUTPUT_COUNT,
                SAMPLES,
            );
            report_direct_tile::<4>(
                "rows",
                distribution,
                &row_source,
                row_weights,
                ROW_OUTPUT_COUNT,
                SAMPLES,
            );
            report_direct_tile::<8>(
                "rows",
                distribution,
                &row_source,
                row_weights,
                ROW_OUTPUT_COUNT,
                SAMPLES,
            );
            report_direct_tile::<16>(
                "rows",
                distribution,
                &row_source,
                row_weights,
                ROW_OUTPUT_COUNT,
                SAMPLES,
            );
            let row_direct = report_direct_tile::<32>(
                "rows",
                distribution,
                &row_source,
                row_weights,
                ROW_OUTPUT_COUNT,
                SAMPLES,
            );
            report_direct_tile::<64>(
                "rows",
                distribution,
                &row_source,
                row_weights,
                ROW_OUTPUT_COUNT,
                SAMPLES,
            );
            macro_rules! report_tiles {
                ($shape:literal, $source:expr, $weights:expr, $output_len:expr, $direct:expr) => {
                    report_lb2_tile::<4>(
                        $shape,
                        distribution,
                        $source,
                        $weights,
                        $output_len,
                        $direct,
                        SAMPLES,
                    );
                    report_lb2_tile::<8>(
                        $shape,
                        distribution,
                        $source,
                        $weights,
                        $output_len,
                        $direct,
                        SAMPLES,
                    );
                    report_lb2_tile::<16>(
                        $shape,
                        distribution,
                        $source,
                        $weights,
                        $output_len,
                        $direct,
                        SAMPLES,
                    );
                    report_lb2_tile::<32>(
                        $shape,
                        distribution,
                        $source,
                        $weights,
                        $output_len,
                        $direct,
                        SAMPLES,
                    );
                    report_lb2_tile::<64>(
                        $shape,
                        distribution,
                        $source,
                        $weights,
                        $output_len,
                        $direct,
                        SAMPLES,
                    );
                };
            }
            report_tiles!(
                "columns",
                &column_source,
                &weights,
                COLUMN_OUTPUT_COUNT,
                column_direct
            );
            report_tiles!(
                "rows",
                &row_source,
                row_weights,
                ROW_OUTPUT_COUNT,
                row_direct
            );
        }
    }
}
