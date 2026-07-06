//! Unified witness/relation fold schedule and fused fold-scan entry points.

use super::*;

/// How the current round folds live witness/relation storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FoldRoundKind {
    FlatPair,
    EmbeddedSegmentAxis,
    EmbeddedCoefficientAxis,
}

#[derive(Clone, Copy)]
pub(crate) enum WitnessFoldInput<'a, E: FieldCore> {
    Compact {
        digits: &'a [i8],
        fold_lut: &'a CompactPairFoldLut<E>,
    },
    Full {
        evals: &'a [E],
        challenge: E,
        use_local_view_flat_fold: bool,
    },
}

/// Fused fold-and-scan paths that preserve hot-cache round messages.
pub(crate) enum FusedFoldScan<'a, E: FieldCore> {
    InitialRound2 {
        w_compact: &'a [i8],
        relation_round2: &'a [E],
        r0: E,
        r1: E,
    },
    SegmentAxis {
        w_full: &'a [E],
        challenge: E,
    },
}

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps> AkitaStage2Prover<E> {
    /// Relation and witness storage can fold on different axes in the same round.
    ///
    /// Compact witness digits are still laid out as a flat `live_len` vector until
    /// the first promotion to field evals. Embedded coefficient-axis folding assumes
    /// segment-major field layout, so compact witness must keep flat pairwise fold
    /// even when relation weight uses `EmbeddedCoefficientAxis`.
    pub(super) fn witness_fold_kind(
        relation_kind: FoldRoundKind,
        witness_is_compact: bool,
    ) -> FoldRoundKind {
        if witness_is_compact && relation_kind == FoldRoundKind::EmbeddedCoefficientAxis {
            FoldRoundKind::FlatPair
        } else {
            relation_kind
        }
    }

    pub(super) fn fold_round_kind(&self, folding_segment_round: bool) -> FoldRoundKind {
        if self.in_coefficient_round() && self.use_coefficient_prefix_round() {
            FoldRoundKind::EmbeddedCoefficientAxis
        } else if folding_segment_round && self.use_segment_prefix_round() {
            FoldRoundKind::EmbeddedSegmentAxis
        } else {
            FoldRoundKind::FlatPair
        }
    }

    pub(super) fn fold_witness_flat_compact(
        w_compact: &[i8],
        fold_lut: &CompactPairFoldLut<E>,
    ) -> Vec<E> {
        cfg_into_iter!(0..w_compact.len().div_ceil(2))
            .map(|j| {
                let left = 2 * j;
                let w0 = i16::from(w_compact[left]);
                let w1 = w_compact.get(left + 1).copied().map(i16::from).unwrap_or(0);
                fold_lut.fold(w0, w1)
            })
            .collect()
    }
}

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps + HasOptimizedFold> AkitaStage2Prover<E> {
    pub(super) fn fold_witness_full_owned(
        evals: Vec<E>,
        kind: FoldRoundKind,
        live_segments: usize,
        coeff_len: usize,
        challenge: E,
        use_local_view_flat_fold: bool,
    ) -> Vec<E> {
        if use_local_view_flat_fold && kind == FoldRoundKind::FlatPair {
            let mut evals = evals;
            fold_evals_in_place(&mut evals, challenge);
            evals
        } else {
            Self::fold_witness_polynomial(
                WitnessFoldInput::Full {
                    evals: &evals,
                    challenge,
                    use_local_view_flat_fold: false,
                },
                kind,
                live_segments,
                coeff_len,
            )
        }
    }

    pub(super) fn fold_witness_polynomial(
        input: WitnessFoldInput<'_, E>,
        kind: FoldRoundKind,
        live_segments: usize,
        coeff_len: usize,
    ) -> Vec<E> {
        match (input, kind) {
            (
                WitnessFoldInput::Compact { digits, fold_lut },
                FoldRoundKind::EmbeddedSegmentAxis,
            ) => Self::fold_witness_embedded_segment_compact(
                digits,
                live_segments,
                coeff_len,
                fold_lut,
            ),
            (
                WitnessFoldInput::Full {
                    evals,
                    challenge,
                    ..
                },
                FoldRoundKind::EmbeddedSegmentAxis,
            ) => Self::fold_witness_embedded_segment_full(evals, live_segments, coeff_len, challenge),
            (
                WitnessFoldInput::Full {
                    evals,
                    challenge,
                    ..
                },
                FoldRoundKind::EmbeddedCoefficientAxis,
            ) => Self::fold_witness_embedded_coefficient_full(evals, live_segments, coeff_len, challenge),
            (WitnessFoldInput::Compact { digits, fold_lut }, FoldRoundKind::FlatPair) => {
                Self::fold_witness_flat_compact(digits, fold_lut)
            }
            (
                WitnessFoldInput::Full {
                    evals,
                    challenge,
                    use_local_view_flat_fold,
                },
                FoldRoundKind::FlatPair,
            ) => {
                if use_local_view_flat_fold {
                    let mut out = evals.to_vec();
                    fold_evals_in_place(&mut out, challenge);
                    out
                } else {
                    fold_live_evals_zero_padded(evals, challenge)
                }
            }
            (
                WitnessFoldInput::Compact { .. },
                FoldRoundKind::EmbeddedCoefficientAxis,
            ) => unreachable!("coefficient-axis fold expects full witness storage"),
        }
    }

    pub(super) fn run_fused_fold_scan(
        &self,
        fused: FusedFoldScan<'_, E>,
    ) -> (Vec<E>, NormRoundTerms<E>, [E; 3]) {
        match fused {
            FusedFoldScan::InitialRound2 {
                w_compact,
                relation_round2,
                r0,
                r1,
            } => self.fused_fold_scan_initial_round2(w_compact, relation_round2, r0, r1),
            FusedFoldScan::SegmentAxis { w_full, challenge } => {
                self.fused_fold_scan_segment_axis(w_full, challenge)
            }
        }
    }
}
