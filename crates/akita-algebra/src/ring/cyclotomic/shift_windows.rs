use super::*;
use jolt_field::WithCommitAccumulator;

/// Destination coefficients per register tile. Four `Fp128` tile sums take
/// eight SSE2 or NEON registers or four AVX2 registers, leaving room for the
/// window loads.
const TILE: usize = 4;

/// Every negacyclic shift of one ring element, prepared for batched
/// accumulation into wide destinations.
///
/// Holds the [`WithCommitAccumulator::CommitLanes`] of
/// `[-a_0, …, -a_{D-1}, a_0, …, a_{D-1}]`, so coefficient `j` of `a · X^k` is
/// entry `D + j - k` for every `k < D`. A batch of shifts into one
/// destination then reads each shift as a contiguous window of half-width
/// lanes, sums the batch in registers, and writes each destination tile
/// once, instead of one full read-modify-write of the destination per shift.
/// Loading costs two lane splits per coefficient, so a load pays off when the
/// element is accumulated into several destinations or with several shifts.
#[derive(Debug, Clone)]
pub struct NegacyclicShiftWindows<F: WithCommitAccumulator, const D: usize> {
    lanes: Vec<F::CommitLanes>,
}

impl<F: WithCommitAccumulator, const D: usize> Default for NegacyclicShiftWindows<F, D> {
    fn default() -> Self {
        const { assert!(D.is_multiple_of(TILE)) };
        Self {
            lanes: vec![F::CommitLanes::default(); 2 * D],
        }
    }
}

impl<F: WithCommitAccumulator, const D: usize> NegacyclicShiftWindows<F, D> {
    /// Replaces the held element with `src`.
    #[inline]
    pub fn load(&mut self, src: &CyclotomicRing<F, D>) {
        let (negative, positive) = self.lanes.split_at_mut(D);
        for ((negative, positive), &value) in negative.iter_mut().zip(positive).zip(&src.coeffs) {
            *negative = F::CommitLanes::from(-value);
            *positive = F::CommitLanes::from(value);
        }
    }

    /// `dst += a · Σ_{k ∈ shifts} X^k` for the held element `a`.
    ///
    /// Requires every `k < D`; shifts may repeat and come in any order. Each
    /// shift is one addition against
    /// [`WithCommitAccumulator::MAX_COMMIT_ACCUMULATIONS`] for `dst`.
    #[inline]
    pub fn accumulate_shifts_into(
        &self,
        dst: &mut WideCyclotomicRing<F::Wide, D>,
        shifts: &[usize],
    ) {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: runtime feature detection guarantees AVX2.
            unsafe { self.accumulate_shifts_into_avx2(dst, shifts) };
            return;
        }
        self.accumulate_shifts_into_portable(dst, shifts);
    }

    /// Baseline x86-64 widens `u16` lanes with SSE2 unpacks; AVX2 widens and
    /// adds a full tile row per instruction pair.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn accumulate_shifts_into_avx2(
        &self,
        dst: &mut WideCyclotomicRing<F::Wide, D>,
        shifts: &[usize],
    ) {
        self.accumulate_shifts_into_portable(dst, shifts);
    }

    #[inline(always)]
    fn accumulate_shifts_into_portable(
        &self,
        dst: &mut WideCyclotomicRing<F::Wide, D>,
        shifts: &[usize],
    ) {
        debug_assert!(shifts.iter().all(|&shift| shift < D));
        for (tile, out) in dst.coeffs.chunks_exact_mut(TILE).enumerate() {
            let base = D + tile * TILE;
            let mut sums = [F::Wide::zero(); TILE];
            for &shift in shifts {
                for (sum, &lanes) in sums.iter_mut().zip(&self.lanes[base - shift..][..TILE]) {
                    F::add_commit_lanes(sum, lanes);
                }
            }
            for (out, sum) in out.iter_mut().zip(sums) {
                *out += sum;
            }
        }
    }
}
