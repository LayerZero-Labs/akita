//! Lane rounds with the relation weight merged into one lane table.
//!
//! Once every coefficient variable is bound, the ordinary relation weight
//! (`alpha(r_c) * lw(x)`, or the reduced dense weight) and the structured
//! linear weight `L(r_c, x)` both depend on the lane only. They merge into one
//! table `q`, so the relation message of a lane round is
//! `sum_j (w0 + X dw) (q0 + X dq)` over lane pairs `j`. Each round folds `w`
//! and `q` by its challenge and accumulates the next round's range-image and
//! relation terms in the same pass.

use super::*;
use std::ops::Range;

pub(super) struct LaneProduct<E: Field> {
    /// `q` on lanes `0..weights.len()`, zero beyond.
    weights: Vec<E>,
    witness_scratch: Vec<E>,
    weight_scratch: Vec<E>,
}

impl<E: Field> LaneProduct<E> {
    /// Relation weight at the fully bound lane point.
    pub(super) fn final_weight(&self) -> Result<E, AkitaError> {
        match self.weights.as_slice() {
            [weight] => Ok(*weight),
            _ => Err(AkitaError::InvalidProof),
        }
    }
}

/// Tables read by one lane round.
#[derive(Clone, Copy)]
struct LaneTables<'a, E> {
    witness: &'a [E],
    weights: &'a [E],
    eq_low: &'a [E],
    eq_high: &'a [E],
}

#[inline(always)]
fn fold_pair<E: Fold>(values: &[E], left: usize, fold: &E::Ctx) -> E {
    let even = values.get(left).copied().unwrap_or_else(E::zero);
    let odd = values.get(left + 1).copied().unwrap_or_else(E::zero);
    E::fold_one(fold, even, odd)
}

/// Resize `buffer` to `len`, keeping its allocation when it shrinks.
fn reuse_buffer<E: Field>(buffer: &mut Vec<E>, len: usize) {
    if buffer.len() >= len {
        buffer.truncate(len);
    } else {
        buffer.resize(len, E::zero());
    }
}

/// Range-image and relation terms of lane pairs `pairs` over `live` witness
/// entries. With `FOLD`, entry `i` is the fold of entries `2i` and `2i + 1`
/// of the tables by `fold`, and the folded pairs are written to the outputs,
/// which start at pair `pairs.start`; otherwise entries are read directly.
fn lane_product_tile<E, const FOLD: bool, const SKIP_LINEAR: bool>(
    tables: LaneTables<'_, E>,
    fold: &E::Ctx,
    pairs: Range<usize>,
    live: usize,
    witness_out: &mut [E],
    weight_out: &mut [E],
) -> ([E; 3], [E; 3])
where
    E: Field + Unreduced + Fold,
{
    let entry = |values: &[E], index: usize| {
        if FOLD {
            fold_pair(values, 2 * index, fold)
        } else {
            values.get(index).copied().unwrap_or_else(E::zero)
        }
    };
    let low_bits = tables.eq_low.len().trailing_zeros();
    let low_mask = tables.eq_low.len() - 1;
    let mut norm = [E::zero(); 3];
    let mut relation = [ProductSum::<E>::zero(); 3];
    let mut block_start = pairs.start;
    while block_start < pairs.end {
        let high = block_start >> low_bits;
        let block_end = ((high + 1) << low_bits).min(pairs.end);
        let mut inner = [ProductSum::<E>::zero(); 3];
        for pair in block_start..block_end {
            let (left, right) = (2 * pair, 2 * pair + 1);
            let w0 = entry(tables.witness, left);
            let w1 = if right < live {
                entry(tables.witness, right)
            } else {
                E::zero()
            };
            let q0 = entry(tables.weights, left);
            let q1 = entry(tables.weights, right);
            if FOLD {
                let local = left - 2 * pairs.start;
                witness_out[local] = w0;
                if right < live {
                    witness_out[local + 1] = w1;
                }
                weight_out[local] = q0;
                weight_out[local + 1] = q1;
            }
            let dw = w1 - w0;
            let e_in = tables.eq_low[pair & low_mask];
            inner[0].add(e_in, w0.square() + w0);
            if !SKIP_LINEAR {
                inner[1].add(e_in, dw * (w0 + w0 + E::one()));
            }
            inner[2].add(e_in, dw.square());
            relation[0].add(w0, q0);
            relation[1].add(w1, q1);
            relation[2].add(dw, q1 - q0);
        }
        let e_out = tables.eq_high[high];
        for (norm, inner) in norm.iter_mut().zip(inner) {
            *norm += e_out * inner.finish();
        }
        block_start = block_end;
    }
    // The linear coefficient is `p(1) - p(0) - p_2`.
    let [constant, at_one, quadratic] = relation.map(ProductSum::finish);
    (norm, [constant, at_one - constant - quadratic, quadratic])
}

/// [`lane_product_tile`] with its const parameters chosen at run time.
fn run_lane_product_tile<E: Field + Unreduced + Fold>(
    tables: LaneTables<'_, E>,
    fold: Option<&E::Ctx>,
    skip_linear: bool,
    pairs: Range<usize>,
    live: usize,
    witness_out: &mut [E],
    weight_out: &mut [E],
) -> ([E; 3], [E; 3]) {
    let unused = E::precompute(E::zero());
    match (fold, skip_linear) {
        (Some(fold), true) => {
            lane_product_tile::<E, true, true>(tables, fold, pairs, live, witness_out, weight_out)
        }
        (Some(fold), false) => {
            lane_product_tile::<E, true, false>(tables, fold, pairs, live, witness_out, weight_out)
        }
        (None, true) => lane_product_tile::<E, false, true>(
            tables,
            &unused,
            pairs,
            live,
            witness_out,
            weight_out,
        ),
        (None, false) => lane_product_tile::<E, false, false>(
            tables,
            &unused,
            pairs,
            live,
            witness_out,
            weight_out,
        ),
    }
}

const TAIL_FOLD_CHUNK: usize = 1 << 14;

impl<E: Field + Unreduced + Fold> LaneProduct<E> {
    /// Terms of the round over `witness`, after folding `witness` and the
    /// weights by `challenge` when one is given. With no `eq` tables (the
    /// final fold) nothing follows and `None` is returned.
    fn round_terms(
        &mut self,
        witness: &mut Vec<E>,
        challenge: Option<E>,
        eq: Option<(&[E], &[E])>,
        skip_linear: bool,
    ) -> Option<(NormRoundTerms<E>, [E; 3])> {
        let fold = challenge.map(E::precompute);
        let Some((eq_low, eq_high)) = eq else {
            let fold = fold.expect("the final lane-product fold requires a challenge");
            let fold_all = |values: &[E]| -> Vec<E> {
                (0..values.len().div_ceil(2))
                    .map(|index| fold_pair(values, 2 * index, &fold))
                    .collect()
            };
            *witness = fold_all(witness);
            self.weights = fold_all(&self.weights);
            return None; // The final fold has no following round polynomial.
        };
        let live = if fold.is_some() {
            witness.len().div_ceil(2)
        } else {
            witness.len()
        };
        let pair_count = live.div_ceil(2);
        let tile_pairs = crate::opaque::sumcheck::single_row_tile_pairs(eq_low.len());
        let tables = LaneTables {
            witness: witness.as_slice(),
            weights: self.weights.as_slice(),
            eq_low,
            eq_high,
        };
        let tile = |task: usize, witness_out: &mut [E], weight_out: &mut [E]| {
            let pairs = task * tile_pairs..((task + 1) * tile_pairs).min(pair_count);
            run_lane_product_tile(
                tables,
                fold.as_ref(),
                skip_linear,
                pairs,
                live,
                witness_out,
                weight_out,
            )
        };
        let parts: Vec<([E; 3], [E; 3])> = if let Some(fold) = &fold {
            let weight_len = self.weights.len().div_ceil(2).max(2 * pair_count);
            reuse_buffer(&mut self.witness_scratch, live);
            reuse_buffer(&mut self.weight_scratch, weight_len);
            let (head, tail) = self.weight_scratch.split_at_mut(2 * pair_count);
            cfg_chunks_mut!(tail, TAIL_FOLD_CHUNK)
                .enumerate()
                .for_each(|(chunk, values)| {
                    let first = 2 * pair_count + chunk * TAIL_FOLD_CHUNK;
                    for (offset, value) in values.iter_mut().enumerate() {
                        *value = fold_pair(tables.weights, 2 * (first + offset), fold);
                    }
                });
            cfg_chunks_mut!(self.witness_scratch, 2 * tile_pairs)
                .zip(cfg_chunks_mut!(head, 2 * tile_pairs))
                .enumerate()
                .map(|(task, (witness_out, weight_out))| tile(task, witness_out, weight_out))
                .collect()
        } else {
            cfg_into_iter!(0..pair_count.div_ceil(tile_pairs))
                .map(|task| tile(task, &mut [], &mut []))
                .collect()
        };
        if fold.is_some() {
            mem::swap(witness, &mut self.witness_scratch);
            mem::swap(&mut self.weights, &mut self.weight_scratch);
        }
        let (norm, relation) =
            parts
                .into_iter()
                .fold(([E::zero(); 3], [E::zero(); 3]), |mut totals, part| {
                    add_round_terms(&mut totals, part);
                    totals
                });
        let norm = if skip_linear {
            NormRoundTerms::SkipLinear([norm[0], norm[2]])
        } else {
            NormRoundTerms::Full(norm)
        };
        Some((norm, relation))
    }
}

impl<E: Field + Ring + Unreduced + Fold> RelationRangeImageProver<E> {
    /// Enter the lane-product state once every coefficient variable is bound,
    /// merging the relation weight and the structured linear terms into one
    /// lane table.
    pub(super) fn enter_lane_product(&mut self) {
        // Validate prefix/witness combinations before changing the relation state.
        let has_compact_prefix = self.compact_quotient_prefix().is_some();
        if matches!(self.relation_state, RelationRoundState::LaneProduct(_)) {
            assert!(
                !self.in_coefficient_round(),
                "lane-product state requires bound coefficients"
            );
            assert!(
                matches!(self.witness_state, WitnessState::FoldedSuffix(_)),
                "lane-product state requires a folded witness"
            );
            return; // The lane-product transition already happened.
        }
        if self.in_coefficient_round() {
            assert!(
                has_compact_prefix || self.rounds_completed == 0
                    || matches!(self.witness_state, WitnessState::FoldedSuffix(_)),
                "coefficient rounds outside the compact prefix require a folded witness after round zero"
            );
            return; // Coefficient rounds still use the original relation weights.
        }
        assert!(
            !has_compact_prefix,
            "lane-product transition requires a completed or disabled compact prefix"
        );
        if let WitnessState::CompactPrefix(compact_witness) = &self.witness_state {
            // Only a table without coefficient variables reaches its lane
            // rounds unfolded.
            assert_eq!(
                self.rounds_completed, 0,
                "compact lane witness requires no coefficient rounds"
            );
            let witness = compact_witness
                .view()
                .iter()
                .map(|digit| E::from_i64(i64::from(digit)))
                .collect();
            self.witness_state = WitnessState::FoldedSuffix(witness);
        }
        let entered = RelationRoundState::LaneProduct(LaneProduct {
            weights: Vec::new(),
            witness_scratch: Vec::new(),
            weight_scratch: Vec::new(),
        });
        let (mut weights, weight_scale) = match mem::replace(&mut self.relation_state, entered) {
            RelationRoundState::QuotientFactored { mut weights, .. } => {
                assert_eq!(
                    weights.common_alpha_factor().len(),
                    1,
                    "lane-product transition requires bound coefficient weights"
                );
                let alpha = weights.common_alpha_factor()[0];
                (mem::take(weights.components_mut().1), alpha)
            }
            RelationRoundState::ReducedDense { weights } => (weights.into_evaluations(), E::one()),
            RelationRoundState::LaneProduct(_) => unreachable!("checked above"),
        };
        let live = self.live_lane_count;
        // Only a nonzero weight past the live lanes extends the support.
        let tail = weights.get(live..).unwrap_or_default();
        #[cfg(feature = "parallel")]
        let last_nonzero = tail.par_iter().position_last(|weight| !weight.is_zero());
        #[cfg(not(feature = "parallel"))]
        let last_nonzero = tail.iter().rposition(|weight| !weight.is_zero());
        let support = last_nonzero.map_or(live, |last| live + last + 1);
        weights.resize(support, E::zero());
        let (live_weights, tail) = weights.split_at_mut(live);
        if weight_scale != E::one() {
            cfg_iter_mut!(tail).for_each(|weight| *weight *= weight_scale);
        }
        self.linear_terms
            .drain_into_lane_weights(live_weights, weight_scale);
        if let RelationRoundState::LaneProduct(lane) = &mut self.relation_state {
            lane.weights = weights;
        } else {
            unreachable!("lane-product transition must install lane-product state");
        }
    }

    /// `(combined, range-image)` messages of the current round in the
    /// lane-product state, entering it first once the coefficient block is
    /// bound. Returns `None` outside that state.
    pub(super) fn lane_product_round_polys(
        &mut self,
    ) -> Option<(UnivariatePoly<E>, UnivariatePoly<E>)> {
        self.enter_lane_product();
        if self.in_coefficient_round() {
            return None; // Coefficient rounds are computed by the caller.
        }
        let skip_linear = self.can_skip_norm_linear_coeff();
        let (RelationRoundState::LaneProduct(lane), WitnessState::FoldedSuffix(witness)) =
            (&mut self.relation_state, &mut self.witness_state)
        else {
            unreachable!("lane-product rounds require lane-product weights and a folded witness");
        };
        let (norm, relation) = lane
            .round_terms(
                witness,
                None,
                Some(self.split_eq.remaining_eq_tables()),
                skip_linear,
            )
            .expect("a lane-product round with equality tables produces round terms");
        let (norm_poly, relation_poly) = self.polys_from_terms(norm, relation);
        Some((self.combine_polys(&norm_poly, &relation_poly), norm_poly))
    }

    /// Bind `r` in the lane-product state: fold the witness and the lane
    /// table and cache the next round's message.
    pub(super) fn ingest_lane_product_challenge(&mut self, r: E) {
        self.split_eq.bind(r);
        let last = self.rounds_completed + 1 == self.num_vars;
        let skip_linear = !last && self.can_skip_norm_linear_coeff();
        let (RelationRoundState::LaneProduct(lane), WitnessState::FoldedSuffix(witness)) =
            (&mut self.relation_state, &mut self.witness_state)
        else {
            unreachable!(
                "lane-product ingestion requires lane-product weights and a folded witness"
            );
        };
        let eq = (!last).then(|| self.split_eq.remaining_eq_tables());
        let terms = lane.round_terms(witness, Some(r), eq, skip_linear);
        self.live_lane_count = self.live_lane_count.div_ceil(2);
        if let Some((norm, relation)) = terms {
            self.cached_round_poly = Some(self.combine_terms(norm, relation));
        }
    }
}
