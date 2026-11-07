//! Root address translations of an immutable, locally packed typed prefix.

use super::*;
use crate::root_repacking::RootCoefficientRepacking;

/// A validated view of local setup roles in the common outgoing witness.
/// Local identities, setup labels, column numbers and prefix cells never move.
/// The global repacking is the source of every translated segment start.
#[derive(Clone, Debug)]
pub struct BinaryPrefixTranslation<'a> {
    local: &'a BinaryTypedPrefix,
    starts: Vec<usize>,
    response_start: usize,
    root_domain_len: usize,
}

impl BinaryTypedPrefix {
    /// Match every local segment to a segment of the common root layout.
    /// Unlike the singleton's local lanes, translated starts need not be
    /// stride aligned. The selected natural and padded lanes must fit W.
    pub fn translated<'a>(
        &'a self,
        manifest: &BinaryCommittedGroupManifest,
        plan: &BinaryRootWitnessPlan,
        root: &RootCoefficientRepacking,
        segment_indices: &[usize],
    ) -> Result<BinaryPrefixTranslation<'a>, AkitaError> {
        if root.domain_len().trailing_zeros() > 32
            || segment_indices.len() != plan.repacking().segments().len()
            || segment_indices.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(AkitaError::InvalidInput(
                "invalid typed prefix translation".into(),
            ));
        }
        for (local, &global) in plan.repacking().segments().iter().zip(segment_indices) {
            if root.segments().get(global) != Some(local) {
                return Err(AkitaError::InvalidInput(
                    "translated prefix segment shape mismatch".into(),
                ));
            }
        }
        if *self != Self::from_plan(manifest, plan)? {
            return Err(AkitaError::InvalidInput(
                "translated prefix local layout mismatch".into(),
            ));
        }
        let response_start = root
            .segment_range(*segment_indices.first().ok_or(AkitaError::InvalidProof)?)?
            .start;
        let mut starts = Vec::new();
        for segment in std::iter::once(1).chain(plan.opening_segments()) {
            let global = *segment_indices
                .get(segment)
                .ok_or(AkitaError::InvalidProof)?;
            let start = root.segment_range(global)?.start;
            starts.push(start);
        }
        if starts.len() != self.segments.len() {
            return Err(AkitaError::InvalidInput(
                "translated prefix local layout mismatch".into(),
            ));
        }
        Ok(BinaryPrefixTranslation {
            local: self,
            starts,
            response_start,
            root_domain_len: root.domain_len(),
        })
    }
}

impl BinaryPrefixTranslation<'_> {
    /// Address of the first response digit, for the translated A product.
    pub fn response_start(&self) -> usize {
        self.response_start
    }

    /// Live block representation of F on the unchanged local prefix domain.
    pub fn prover_weight_blocks<E: Field>(
        &self,
        a: &BinaryAProduct<E>,
        root_point: &[E],
        alpha: E,
        batches: [&[E]; 2],
        prefix_len: usize,
    ) -> Result<TypedPrefixWeights<E>, AkitaError> {
        self.validate_point(root_point)?;
        self.local.prover_weight_blocks_at(
            a,
            root_point,
            alpha,
            batches,
            prefix_len,
            Some(&self.starts),
        )
    }

    /// Factored B/D terminal weight, without reading setup coefficients or
    /// expanding the period axis. Addition at a translated start can carry
    /// from coefficient bits into lane bits and from there into period bits.
    pub fn digit_weight<E: Field>(
        &self,
        point: &[E],
        root_point: &[E],
        alpha: E,
        batches: [&[E]; 2],
    ) -> Result<E, AkitaError> {
        self.validate_point(root_point)?;
        let prefix_len = checked::pow2(point.len()).ok_or(AkitaError::InvalidProof)?;
        self.local.validate_prefix(prefix_len)?;
        self.local.validate_batches(batches)?;
        let lambda = ResidueKernelPoint::new(alpha, SLOTS)?.field_kernel(&EqPolynomial::evals(
            point.get(..SLOT_BITS).ok_or(AkitaError::InvalidProof)?,
        )?)?;
        let views = self
            .local
            .regions
            .iter()
            .map(|region| region.view(point))
            .collect::<Result<Vec<_>, _>>()?;
        let mut total = E::zero();
        for (index, (segment, &start)) in self.local.segments.iter().zip(&self.starts).enumerate() {
            let low_bits = segment.lanes.stride().trailing_zeros() as usize
                + segment.period.trailing_zeros() as usize;
            let window = checked::pow2(low_bits).ok_or(AkitaError::InvalidProof)?;
            let low_point = root_point.get(..low_bits).ok_or(AkitaError::InvalidProof)?;
            let high_point = root_point.get(low_bits..).ok_or(AkitaError::InvalidProof)?;
            let low_eq = EqPolynomial::evals(low_point)?;
            for piece in self
                .local
                .pieces
                .iter()
                .filter(|piece| piece.segment == index)
            {
                let (region_factor, coordinates) =
                    views.get(piece.region).ok_or(AkitaError::InvalidProof)?;
                let column_bits =
                    checked::sum([SLOT_BITS, piece.width_bits]).ok_or(AkitaError::InvalidProof)?;
                let column_eq = EqPolynomial::evals(
                    coordinates
                        .get(SLOT_BITS..column_bits)
                        .ok_or(AkitaError::InvalidProof)?,
                )?;
                let period_point = coordinates
                    .get(column_bits..)
                    .ok_or(AkitaError::InvalidProof)?;
                let periods = checked::pow2(piece.period_bits).ok_or(AkitaError::InvalidProof)?;
                let mut grouped: Vec<(usize, usize, E)> = Vec::new();
                for p in 0..segment.period.min(segment.lane_count()) {
                    let live = checked::div_ceil(segment.lane_count() - p, segment.period)
                        .ok_or(AkitaError::InvalidProof)?;
                    let live = live.saturating_sub(piece.q0).min(periods);
                    if live == 0 {
                        continue;
                    }
                    for coefficient in 0..segment.lanes.degree() {
                        let compact =
                            checked::mul_add(p, segment.lanes.degree(), segment.offset % SLOTS)
                                .and_then(|base| checked::sum([base, coefficient]))
                                .ok_or(AkitaError::InvalidProof)?;
                        let Some(column) = (compact / SLOTS)
                            .checked_sub(piece.first)
                            .filter(|&c| c < column_eq.len())
                        else {
                            continue;
                        };
                        let address = checked::mul_add(p, segment.lanes.stride(), start)
                            .and_then(|base| checked::sum([base, coefficient]))
                            .ok_or(AkitaError::InvalidProof)?;
                        let high = checked::sum([address / window, piece.q0])
                            .ok_or(AkitaError::InvalidProof)?;
                        let weight = *low_eq
                            .get(address % window)
                            .ok_or(AkitaError::InvalidProof)?
                            * *lambda
                                .get(compact % SLOTS)
                                .ok_or(AkitaError::InvalidProof)?
                            * *column_eq.get(column).ok_or(AkitaError::InvalidProof)?;
                        if let Some(entry) = grouped
                            .iter_mut()
                            .find(|entry| (entry.0, entry.1) == (high, live))
                        {
                            entry.2 += weight;
                        } else {
                            grouped.push((high, live, weight));
                        }
                    }
                }
                let batch = *batches
                    .get(segment.role)
                    .and_then(|rows| rows.get(piece.row))
                    .ok_or(AkitaError::InvalidProof)?;
                for (high, live, weight) in grouped {
                    total += batch
                        * *region_factor
                        * weight
                        * paired_offset_eq_sum(
                            period_point,
                            piece.base >> column_bits,
                            high_point,
                            high,
                            live,
                        )?;
                }
            }
        }
        Ok(total)
    }

    fn validate_point<E>(&self, root_point: &[E]) -> Result<(), AkitaError> {
        if root_point.len() != self.root_domain_len.trailing_zeros() as usize {
            return Err(AkitaError::InvalidInput(
                "translated prefix root point mismatch".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "typed_prefix_translation_tests.rs"]
mod tests;
