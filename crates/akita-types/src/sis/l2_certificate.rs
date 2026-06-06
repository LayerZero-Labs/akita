//! Public, proof-free L2 folded-witness certificate mode and its pure leaves.
//!
//! The committed-fold A-role can be priced two ways (see [`super::norm_bound`]):
//! against the *deterministic* worst-case squared bound
//! [`l2_bound_squared`](super::norm_bound::l2_bound_squared), or against a
//! *realized* certificate that proves the exact squared norm
//! `Σ z_aug[i]^2 = B_l2` in protocol (sumcheck 1) and ties it to the committed
//! witness (sumcheck 2). A level may carry the realized certificate only when
//! the stage-1 sum-of-squares accumulates in the field *without wrapping* —
//! otherwise the equality `Σ z_aug[i]^2 = B_l2` is meaningless modulo `q`.
//!
//! [`certificate_mode`] is the field-capacity gate that decides this. It is a
//! pure function of public, squared-domain scalars derived from the level
//! schedule/layout/algebra and the accumulation field — never from proof bytes
//! — so prover and verifier compute the same [`L2CertMode`], and the proof
//! shape (claim lists, the `B_l2` wire, the `ell_hat` witness segment) is fixed
//! *before* any verifier-facing scalar is parsed.
//!
//! Every quantity here is squared and exact (`u128`), matching the L2 sizing
//! leaves in [`super::norm_bound`]: the realized squared norm `B_l2`, the
//! deterministic bound `l2_bound_squared`, and the field modulus `q_eff` are all
//! squared-domain integers. These are pure leaves; callers (planner, prover,
//! verifier) wire them to per-level parameters explicitly, the same way the
//! existing SIS leaves are composed.

use akita_field::AkitaError;

/// Whether a fold level carries the realized L2 certificate.
///
/// Computed identically by prover and verifier via [`certificate_mode`]; it is
/// never read from proof bytes, so the proof shape is fixed before any
/// verifier-facing scalar is parsed. The mode controls whether `B_l2` is
/// present, whether `ell_hat` has nonzero length, whether stage 1 fuses the L2
/// claim into its root, whether stage 2 includes the virtualization claim, and
/// which claim order is bound in the transcript descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L2CertMode {
    /// No realized L2 certificate. The level prices the A-role at the
    /// deterministic worst-case bound `l2_bound_squared` and keeps the existing
    /// eq-factored stage-1 range-check proof.
    Deterministic,
    /// The level carries the L2 certificate: sumcheck 1 (`Σ z_aug[i]^2 = B_l2`)
    /// and sumcheck 2 (`z_aug = G' · w_next`) are active, with
    /// `B_l2 <= l2_bound_squared`.
    Realized,
}

/// Public, squared-domain inputs to the [`certificate_mode`] field-capacity
/// gate.
///
/// All fields are exact `u128` integers in the squared domain. The caller
/// derives them from the level's public parameters (schedule/layout/algebra and
/// the stage-1 accumulation field); no proof bytes are read. Keeping the gate a
/// leaf over explicit scalars — rather than reaching into per-level structs —
/// mirrors how the other SIS leaves in this module are composed and keeps the
/// no-wrap decision trivially testable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L2GateInputs {
    /// Number of certified coefficients `coeffs = W · D` (folded ring rows `W`
    /// times ring dimension `D`): the count of squares the stage-1 sum
    /// accumulates over the augmented witness `z_aug`.
    pub coeffs: u128,
    /// Verifier-assertable per-coefficient magnitude bound on `|z_aug[i]|`: the
    /// largest balanced base-`b` digit the stage-1 range check admits,
    /// `balanced_digit_max(log_basis, num_digits_fold)`.
    pub per_coeff_bound: u128,
    /// Deterministic worst-case squared bound
    /// [`l2_bound_squared`](super::norm_bound::l2_bound_squared). The gate is
    /// evaluated here, not at the prover-chosen wire `B_l2`; since any
    /// admissible `B_l2` never exceeds it, capacity at this bound implies
    /// capacity for every admissible certificate, which keeps the mode decision
    /// non-circular with respect to the proof payload.
    pub l2_bound_squared: u128,
    /// Modulus `q_eff` of the field the stage-1 sum-of-squares accumulates in.
    pub q_eff: u128,
}

/// Structural worst-case value of the stage-1 sum-of-squares accumulation,
/// including the four-square slack headroom, or `None` on `u128` overflow.
///
/// ```text
/// coeffs · per_coeff_bound^2  +  4 · l2_bound_squared
/// ```
///
/// The first term upper-bounds `Σ z_aug[i]^2` from the range-checked digits;
/// the `4 · l2_bound_squared` headroom covers the four slack squares of the
/// Lagrange equality `Σ z_aug[i]^2 + Σ_j ell_j^2 = B_l2`, where each
/// `ell_j^2 <= B_l2 <= l2_bound_squared`.
#[inline]
fn structural_square_sum_upper_bound(inputs: &L2GateInputs) -> Option<u128> {
    let per_coeff_sq = inputs.per_coeff_bound.checked_mul(inputs.per_coeff_bound)?;
    let range_term = per_coeff_sq.checked_mul(inputs.coeffs)?;
    let slack_headroom = inputs.l2_bound_squared.checked_mul(4)?;
    range_term.checked_add(slack_headroom)
}

/// Public, proof-free field-capacity gate deciding a level's [`L2CertMode`].
///
/// A level may carry the realized certificate only when the structural
/// worst-case squared accumulation fits the field without wrapping:
///
/// ```text
/// coeffs · per_coeff_bound^2  +  4 · l2_bound_squared  <  q_eff.
/// ```
///
/// Evaluating at `l2_bound_squared` (not the wire `B_l2`) keeps the decision a
/// pure function of public parameters, so prover and verifier agree on proof
/// shape before any certificate byte is read.
///
/// The check is total: if the left side overflows `u128` it cannot fit any
/// field modulus, so the gate fails closed to [`L2CertMode::Deterministic`] —
/// always the sound fallback.
#[must_use]
pub fn certificate_mode(inputs: &L2GateInputs) -> L2CertMode {
    match structural_square_sum_upper_bound(inputs) {
        Some(lhs) if lhs < inputs.q_eff => L2CertMode::Realized,
        _ => L2CertMode::Deterministic,
    }
}

/// First-cut realized-bucket policy: the smallest power of two `>= z_squared`,
/// capped by the deterministic bound `l2_bound_squared`.
///
/// Returns `0` for the degenerate all-zero statement (`z_squared == 0`); the
/// sumcheck equality then forces every `z_aug` entry to zero. This is a
/// self-contained protocol-construction seam, not rank pricing: it lets the
/// prover construct sumcheck 1 with a concrete `B_l2` satisfying
/// `Z_SQUARED <= B_l2 <= l2_bound_squared` without depending on the future
/// audited L2 SIS ladder, which only *tightens* the binding rank and never
/// changes this correctness contract.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when no power of two `>= z_squared` fits
/// `u128`, or when the chosen bucket exceeds `l2_bound_squared` (the realized
/// squared norm cannot be certified within the deterministic bound).
pub fn select_b_l2(z_squared: u128, l2_bound_squared: u128) -> Result<u128, AkitaError> {
    if z_squared == 0 {
        return Ok(0);
    }
    let bucket = z_squared.checked_next_power_of_two().ok_or_else(|| {
        AkitaError::InvalidSetup("select_b_l2: no power of two >= Z_SQUARED fits u128".to_string())
    })?;
    if bucket > l2_bound_squared {
        return Err(AkitaError::InvalidSetup(format!(
            "select_b_l2: bucket {bucket} exceeds deterministic bound {l2_bound_squared}"
        )));
    }
    Ok(bucket)
}

/// Verifier-side validation of the realized-certificate wire scalar `B_l2`:
/// reject any bucket exceeding the public deterministic bound.
///
/// The verifier derives `l2_bound_squared` from public parameters and rejects a
/// proof whose `B_l2` claims a squared norm larger than the deterministic worst
/// case. `B_l2 == 0` is admitted here (the degenerate all-zero statement); the
/// stage-1 sumcheck, not this bound check, enforces that the witness is actually
/// zero. The lower bound `Z_SQUARED <= B_l2` is enforced by the four-square
/// equality of sumcheck 1, not here.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidProof`] when `b_l2 > l2_bound_squared`.
pub fn validate_realized_bucket(b_l2: u128, l2_bound_squared: u128) -> Result<(), AkitaError> {
    if b_l2 > l2_bound_squared {
        return Err(AkitaError::InvalidProof);
    }
    Ok(())
}

/// A stage-1 sumcheck claim.
///
/// [`Stage1Claim::Range`] (the parent-spec `norm_claim`: the balanced-digit
/// range check on the committed witness) is always present.
/// [`Stage1Claim::L2`] (the parent-spec `l2_claim`: the realized sum-of-squares
/// `Σ z_aug[i]^2 = B_l2`) is fused into the stage-1 root only for
/// [`L2CertMode::Realized`] levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage1Claim {
    /// Balanced-digit range check on the committed witness (eq-factored tree);
    /// the parent-spec `norm_claim`.
    Range,
    /// Realized L2 sum-of-squares `Σ z_aug[i]^2 = B_l2`; the parent-spec
    /// `l2_claim`.
    L2,
}

/// A stage-2 sumcheck claim.
///
/// [`Stage2Claim::S`] (committed-witness `s` opening) and
/// [`Stage2Claim::Relation`] (gadget-recomposition relation) are always
/// present. [`Stage2Claim::Virtualization`] (the parent-spec
/// `virtualization_claim` tying `z_aug = G' · w_next` to the committed witness)
/// is added only for [`L2CertMode::Realized`] levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage2Claim {
    /// Committed-witness `s` opening.
    S,
    /// Gadget-recomposition relation.
    Relation,
    /// `z_aug = G' · w_next` virtualization (realized certificate only); the
    /// parent-spec `virtualization_claim`.
    Virtualization,
}

const STAGE1_DETERMINISTIC: [Stage1Claim; 1] = [Stage1Claim::Range];
const STAGE1_REALIZED: [Stage1Claim; 2] = [Stage1Claim::Range, Stage1Claim::L2];
const STAGE2_DETERMINISTIC: [Stage2Claim; 2] = [Stage2Claim::S, Stage2Claim::Relation];
const STAGE2_REALIZED: [Stage2Claim; 3] = [
    Stage2Claim::S,
    Stage2Claim::Relation,
    Stage2Claim::Virtualization,
];

/// Stage-1 claims carried for `mode`, in canonical batch order.
///
/// The transcript descriptor binds the number and order of these claims; prover
/// and verifier must derive the batching vector from this exact sequence.
#[must_use]
pub fn stage1_claims(mode: L2CertMode) -> &'static [Stage1Claim] {
    match mode {
        L2CertMode::Deterministic => &STAGE1_DETERMINISTIC,
        L2CertMode::Realized => &STAGE1_REALIZED,
    }
}

/// Stage-2 claims carried for `mode`, in canonical batch order.
///
/// The transcript descriptor binds the number and order of these claims; prover
/// and verifier must derive the batching vector from this exact sequence.
#[must_use]
pub fn stage2_claims(mode: L2CertMode) -> &'static [Stage2Claim] {
    match mode {
        L2CertMode::Deterministic => &STAGE2_DETERMINISTIC,
        L2CertMode::Realized => &STAGE2_REALIZED,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate(
        coeffs: u128,
        per_coeff_bound: u128,
        l2_bound_squared: u128,
        q_eff: u128,
    ) -> L2CertMode {
        certificate_mode(&L2GateInputs {
            coeffs,
            per_coeff_bound,
            l2_bound_squared,
            q_eff,
        })
    }

    #[test]
    fn certificate_mode_realized_when_capacity_fits() {
        // coeffs·bound^2 + 4·l2 = 64·16 + 4·100 = 1024 + 400 = 1424 < q.
        assert_eq!(gate(64, 4, 100, 1425), L2CertMode::Realized);
    }

    #[test]
    fn certificate_mode_deterministic_when_at_or_over_capacity() {
        // Equality is not strictly less than q: 1424 < 1424 is false.
        assert_eq!(gate(64, 4, 100, 1424), L2CertMode::Deterministic);
        // Strictly over capacity.
        assert_eq!(gate(64, 4, 100, 1000), L2CertMode::Deterministic);
    }

    #[test]
    fn certificate_mode_fails_closed_on_overflow() {
        // per_coeff_bound^2 overflows u128 -> structural bound is unrepresentable
        // -> cannot fit any modulus -> Deterministic.
        assert_eq!(gate(1, u128::MAX, 0, u128::MAX), L2CertMode::Deterministic);
        // range_term overflows when multiplied by coeffs.
        assert_eq!(
            gate(u128::MAX, 1 << 100, 0, u128::MAX),
            L2CertMode::Deterministic
        );
        // slack headroom (4·l2) overflows.
        assert_eq!(gate(1, 1, u128::MAX, u128::MAX), L2CertMode::Deterministic);
    }

    #[test]
    fn certificate_mode_zero_witness_is_realized_under_any_positive_modulus() {
        // coeffs = 0 -> structural bound is just 4·l2; fits for a large modulus.
        assert_eq!(gate(0, 0, 0, 1), L2CertMode::Realized);
        assert_eq!(gate(0, 1_000, 25, 101), L2CertMode::Realized);
    }

    #[test]
    fn select_b_l2_zero_is_degenerate_bucket() {
        assert_eq!(select_b_l2(0, 0).unwrap(), 0);
        assert_eq!(select_b_l2(0, 1_000).unwrap(), 0);
    }

    #[test]
    fn select_b_l2_rounds_up_to_next_power_of_two() {
        assert_eq!(select_b_l2(1, 1_000).unwrap(), 1);
        assert_eq!(select_b_l2(3, 1_000).unwrap(), 4);
        assert_eq!(select_b_l2(5, 1_000).unwrap(), 8);
        assert_eq!(select_b_l2(1_000, 1_024).unwrap(), 1_024);
    }

    #[test]
    fn select_b_l2_exact_power_of_two_is_unchanged() {
        assert_eq!(select_b_l2(64, 64).unwrap(), 64);
        assert_eq!(select_b_l2(1 << 40, 1 << 40).unwrap(), 1 << 40);
    }

    #[test]
    fn select_b_l2_rejects_bucket_above_bound() {
        // next_pow2(5) = 8 > 7.
        assert!(select_b_l2(5, 7).is_err());
        // next_pow2(1000) = 1024 > 1023.
        assert!(select_b_l2(1_000, 1_023).is_err());
    }

    #[test]
    fn select_b_l2_rejects_overflowing_bucket() {
        // No power of two >= (2^127 + 1) fits u128.
        assert!(select_b_l2((1u128 << 127) + 1, u128::MAX).is_err());
        // 2^127 itself is representable as a power of two.
        assert_eq!(select_b_l2(1u128 << 127, u128::MAX).unwrap(), 1u128 << 127);
    }

    #[test]
    fn validate_realized_bucket_accepts_within_bound() {
        assert!(validate_realized_bucket(0, 0).is_ok());
        assert!(validate_realized_bucket(0, 1_000).is_ok());
        assert!(validate_realized_bucket(512, 1_000).is_ok());
        // Equality at the bound is accepted.
        assert!(validate_realized_bucket(1_000, 1_000).is_ok());
    }

    #[test]
    fn validate_realized_bucket_rejects_above_bound() {
        assert_eq!(
            validate_realized_bucket(1_001, 1_000),
            Err(AkitaError::InvalidProof)
        );
        assert_eq!(
            validate_realized_bucket(1, 0),
            Err(AkitaError::InvalidProof)
        );
    }

    #[test]
    fn stage1_claim_lists_match_mode() {
        assert_eq!(
            stage1_claims(L2CertMode::Deterministic),
            &[Stage1Claim::Range]
        );
        assert_eq!(
            stage1_claims(L2CertMode::Realized),
            &[Stage1Claim::Range, Stage1Claim::L2]
        );
    }

    #[test]
    fn stage2_claim_lists_match_mode() {
        assert_eq!(
            stage2_claims(L2CertMode::Deterministic),
            &[Stage2Claim::S, Stage2Claim::Relation]
        );
        assert_eq!(
            stage2_claims(L2CertMode::Realized),
            &[
                Stage2Claim::S,
                Stage2Claim::Relation,
                Stage2Claim::Virtualization
            ]
        );
    }

    #[test]
    fn realized_claim_lists_extend_deterministic_in_order() {
        // The realized lists are the deterministic lists with the certificate
        // claims appended, never reordered: the shared prefix must match.
        let s1_det = stage1_claims(L2CertMode::Deterministic);
        let s1_real = stage1_claims(L2CertMode::Realized);
        assert_eq!(&s1_real[..s1_det.len()], s1_det);

        let s2_det = stage2_claims(L2CertMode::Deterministic);
        let s2_real = stage2_claims(L2CertMode::Realized);
        assert_eq!(&s2_real[..s2_det.len()], s2_det);
    }
}
