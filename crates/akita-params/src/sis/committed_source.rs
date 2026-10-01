//! The producer contract a committed source must satisfy.
//!
//! A schedule prices its root from two independent declarations:
//!
//! * the **class** — what shape the source has, which decides the honest-fold
//!   sizing rule and, for a unit one-hot source, the per-chunk sparsity its
//!   response caps assume;
//! * the **bound** — how wide a centered coefficient may be
//!   ([`DecompositionParams::log_commit_bound`]), which decides the A-role digit
//!   depth and how much range the final digit plane is charged.
//!
//! They are deliberately independent: a preset declares them separately, and
//! neither can be inferred from the other. In particular `log_commit_bound == 1`
//! is **not** a test for "is this one-hot" — that inference is exactly what the
//! bounded-source work removed from the config macro, and re-deriving it here
//! would put it back.
//!
//! [`CommittedSourceContract`] carries both so that admission, pricing, and the
//! generated-table annotations all read one value instead of reconstructing a
//! partial view of it. Producing a source that satisfies the interval but not the
//! class is not a proof-soundness break — the verifier still enforces its frozen
//! caps — but it is an unsupported commitment whose completeness and grinding
//! budget no longer follow from the planner model, so it must be refused at the
//! producer boundary rather than discovered later as a proof failure.

use super::decomposition_digits::checked_balanced_digit_representable_bounds;
use super::honest_fold_policy::{
    BalancedSignedDigitFoldPolicy, HonestFoldPolicySpec, UnitOneHotFoldPolicy,
};
use super::{onehot_source::canonical_source_classes, FoldWitnessNorms};
use crate::DecompositionParams;
use akita_error::AkitaError;

/// What a committed source must **be**, independent of how wide its
/// coefficients are.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CommittedSourceClass {
    /// Structurally `{0, 1}`, with at most one hot position per
    /// `source_chunk_size` source coefficients.
    ///
    /// The one-hot response model prices per-chunk sparsity, not just coefficient
    /// magnitude, so a dense source whose values happen to be `0`/`1` is **not**
    /// admissible here even though it fits every magnitude interval.
    UnitOneHot {
        /// Logical source coefficients per hot position.
        source_chunk_size: usize,
    },
    /// Balanced signed digits over the declared bound.
    ///
    /// Admissibility is a magnitude question only, so it is decided by
    /// [`CommittedSourceContract::accepted_bounds`].
    BalancedSignedDigit,
}

impl CommittedSourceClass {
    /// Project this source class onto its offline honest-fold sizing policy.
    ///
    /// The class remains the declaration. Field width is geometry needed only
    /// by the sizing policy.
    #[must_use]
    pub const fn honest_fold_policy(self, field_bits: u32) -> HonestFoldPolicySpec {
        match self {
            Self::UnitOneHot { source_chunk_size } => HonestFoldPolicySpec::UnitOneHot(
                UnitOneHotFoldPolicy::canonical(field_bits, source_chunk_size),
            ),
            Self::BalancedSignedDigit => HonestFoldPolicySpec::BalancedSignedDigit(
                BalancedSignedDigitFoldPolicy::universal(field_bits),
            ),
        }
    }

    /// Chunk size a source representation must report to satisfy this class, or
    /// `None` when the class imposes no structural requirement.
    ///
    /// Only the unit one-hot class is structurally restrictive. A one-hot source
    /// committed under a balanced-digit schedule is admissible: its digit energy
    /// is strictly below what that schedule charges, so the pricing stays
    /// conservative.
    #[must_use]
    pub const fn required_onehot_chunk_size(self) -> Option<usize> {
        match self {
            Self::UnitOneHot { source_chunk_size } => Some(source_chunk_size),
            Self::BalancedSignedDigit => None,
        }
    }
}

/// The complete producer contract: what the source must be, and how wide its
/// centered coefficients may be.
///
/// Construction is validated and the fields are private, so every accessor below
/// operates on a decomposition that already passed
/// [`DecompositionParams::validate`]. A type named "contract" must not admit the
/// invalid states its own methods assume away: `declared_bounds` computes
/// `log_commit_bound - 1`, which on a zero bound would panic in debug and wrap in
/// release — silently reporting an invalid declaration as *unconstrained*, the
/// most permissive answer possible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommittedSourceContract {
    class: CommittedSourceClass,
    decomposition: DecompositionParams,
}

/// Unrounded norms of a source under its selected inner decomposition.
#[derive(Clone, Copy, Debug)]
pub struct SourceNorms {
    /// Worst digit-plane ring norms used by honest fold sizing.
    pub fold_witness: FoldWitnessNorms,
    /// Upper bound on squared coefficient energy across all source digit planes.
    pub l2_sq: u128,
    /// Maximum squared coefficient in any source digit plane.
    pub coefficient_sq_max: u128,
}

impl CommittedSourceContract {
    /// Build the contract a config declares, rejecting one it cannot honour.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when the decomposition fails
    /// [`DecompositionParams::validate`], or when a unit one-hot class carries a
    /// chunk size that is not a nonzero power of two.
    pub fn try_new(
        class: CommittedSourceClass,
        decomposition: DecompositionParams,
    ) -> Result<Self, AkitaError> {
        decomposition.validate()?;
        if let CommittedSourceClass::UnitOneHot { source_chunk_size } = class {
            if source_chunk_size == 0 || !source_chunk_size.is_power_of_two() {
                return Err(AkitaError::InvalidSetup(format!(
                    "unit one-hot source chunk size {source_chunk_size} must be a nonzero power of two"
                )));
            }
        }
        Ok(Self {
            class,
            decomposition,
        })
    }

    /// Derive both fold-plane norms and aggregate energy from one source contract.
    ///
    /// Balanced planes use the remaining declared bits, with a unit carry
    /// allowance past the bound. Fold sizing maximizes over planes; energy sums
    /// their squares. The selected basis must fit the admitted i8/i16 inner
    /// kernels. No planner bucketing or probability scaling occurs here.
    ///
    /// # Errors
    /// Returns an error for unsupported inner digit geometry, invalid one-hot
    /// length/alignment, or arithmetic overflow.
    pub fn source_norms(
        self,
        log_basis: u32,
        digit_count: usize,
        ring_dimension: usize,
        logical_len: usize,
    ) -> Result<SourceNorms, AkitaError> {
        if log_basis == 0
            || log_basis > crate::MAX_I16_LOG_BASIS
            || digit_count == 0
            || digit_count > self.decomposition.field_bits() as usize + 1
            || ring_dimension == 0
            || logical_len == 0
        {
            return Err(AkitaError::InvalidSetup(
                "invalid committed-source norm geometry".into(),
            ));
        }
        match self.class {
            CommittedSourceClass::BalancedSignedDigit => {
                let mut energy = 0u128;
                let mut maximum = 0u128;
                for plane in 0..digit_count {
                    let consumed = u32::try_from(plane)
                        .ok()
                        .and_then(|plane| plane.checked_mul(log_basis))
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("digit-plane width overflow".into())
                        })?;
                    let bits = self
                        .decomposition
                        .log_commit_bound
                        .saturating_sub(consumed)
                        .min(log_basis)
                        .max(1);
                    let reach = 1u128 << (bits - 1);
                    energy = energy.checked_add(reach * reach).ok_or_else(|| {
                        AkitaError::InvalidSetup("source digit energy overflow".into())
                    })?;
                    maximum = maximum.max(reach);
                }
                Ok(SourceNorms {
                    fold_witness: FoldWitnessNorms::new(
                        maximum,
                        maximum.checked_mul(ring_dimension as u128).ok_or_else(|| {
                            AkitaError::InvalidSetup("source ring norm overflow".into())
                        })?,
                    ),
                    l2_sq: energy
                        .checked_mul(logical_len as u128)
                        .ok_or_else(|| AkitaError::InvalidSetup("source energy overflow".into()))?,
                    coefficient_sq_max: maximum * maximum,
                })
            }
            CommittedSourceClass::UnitOneHot { source_chunk_size } => {
                if digit_count != 1 {
                    return Err(AkitaError::InvalidSetup(
                        "one-hot sources require one digit plane".into(),
                    ));
                }
                let classes = canonical_source_classes(ring_dimension, source_chunk_size)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("unsupported one-hot source geometry".into())
                    })?;
                let infinity = classes
                    .iter()
                    .map(|class| class.infinity_norm())
                    .max()
                    .unwrap_or(1);
                let l1 = classes
                    .iter()
                    .filter_map(|class| class.l1_norm())
                    .max()
                    .unwrap_or(infinity);
                if !logical_len.is_multiple_of(source_chunk_size)
                    || !logical_len.is_multiple_of(ring_dimension)
                {
                    return Err(AkitaError::InvalidSetup(
                        "unsupported one-hot source length".into(),
                    ));
                }
                // Distinct source chunks occupy distinct canonical coefficients;
                // maximize over every permitted hot position within one group.
                let per_group_energy = classes
                    .iter()
                    .map(|class| class.nonzero_count as u128)
                    .max()
                    .unwrap_or(0);
                let group_count = logical_len / source_chunk_size.max(ring_dimension);
                let l2_sq = per_group_energy
                    .checked_mul(group_count as u128)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("one-hot source energy overflow".into())
                    })?;
                let coefficient_sq_max =
                    u128::from(classes.iter().any(|class| class.nonzero_count > 0));
                Ok(SourceNorms {
                    fold_witness: FoldWitnessNorms::new(infinity, l1),
                    l2_sq,
                    coefficient_sq_max,
                })
            }
        }
    }

    /// Declared source class.
    #[must_use]
    pub const fn class(self) -> CommittedSourceClass {
        self.class
    }

    /// Declared decomposition, whose `log_commit_bound` is the source bound.
    #[must_use]
    pub const fn decomposition(self) -> DecompositionParams {
        self.decomposition
    }

    /// Centered interval the **declared bound alone** admits, as
    /// `(negative_abs_reach, positive_reach)` with `None` meaning "beyond every
    /// `u128`, so it cannot be exceeded".
    ///
    /// Following [`DecompositionParams::log_commit_bound`], a bound of `k` signed
    /// bits is `[-2^(k-1), 2^(k-1) - 1]`. Two cases are deliberately
    /// unconstrained:
    ///
    /// * **The full-field endpoint** (`log_commit_bound == field_bits`). Every
    ///   field element is in range by construction, and the decomposition
    ///   switches to asymmetric centering there, so the symmetric interval would
    ///   not describe it.
    /// * **The unit one-hot class.** Its admitted set is structural (`{0, 1}`
    ///   plus per-chunk sparsity), enforced by
    ///   [`CommittedSourceClass::required_onehot_chunk_size`]. It is keyed on the
    ///   *class*, never on `log_commit_bound == 1`: the bound there selects a
    ///   digit depth of one and was never an acceptance interval, and reading it
    ///   as the signed range `[-1, 0]` would reject a hot position.
    #[must_use]
    pub fn declared_bounds(self) -> (Option<u128>, Option<u128>) {
        if matches!(self.class, CommittedSourceClass::UnitOneHot { .. })
            || !self.decomposition.has_bounded_committed_source()
        {
            return (None, None);
        }
        // `k` signed bits is one sign bit plus `k - 1` magnitude bits. `k >= 1`
        // holds because construction ran `DecompositionParams::validate`.
        debug_assert!(self.decomposition.log_commit_bound >= 1);
        let negative_abs = 1u128.checked_shl(self.decomposition.log_commit_bound - 1);
        (negative_abs, negative_abs.map(|reach| reach - 1))
    }

    /// Centered interval a committed source must fit at this level's selected A
    /// geometry, as `(negative_abs_reach, positive_reach)`.
    ///
    /// The **intersection** of two independent constraints:
    ///
    /// 1. **Representability** — the coefficient must be recoverable from
    ///    `num_digits_inner` balanced base-`2^log_basis_inner` digits. Outside it
    ///    the decomposition keeps only the scheduled digits, so the commitment
    ///    binds a truncation.
    /// 2. **Declaration** — [`Self::declared_bounds`], the range the schedule was
    ///    *priced* for.
    ///
    /// The two differ because the depth rounds up: 13 base-`2^5` digits span 65
    /// bits, so they represent about `±2^64` while a `log_commit_bound = 64`
    /// schedule is priced for `±2^63`. The gap reaches 256x at the shipped
    /// `log_basis_inner = 9` geometry, so checking representability alone would
    /// accept coefficients the schedule never declared.
    ///
    /// This is a magnitude test only. A source also has to satisfy
    /// [`Self::class`], which magnitudes cannot express.
    #[must_use]
    pub fn accepted_bounds(
        self,
        log_basis_inner: u32,
        num_digits_inner: usize,
    ) -> (Option<u128>, Option<u128>) {
        let (representable_negative, representable_positive) =
            checked_balanced_digit_representable_bounds(log_basis_inner, num_digits_inner);
        let (declared_negative, declared_positive) = self.declared_bounds();
        (
            tighter_reach(representable_negative, declared_negative),
            tighter_reach(representable_positive, declared_positive),
        )
    }
}

/// Smaller of two reaches, where `None` is "beyond every `u128`" and therefore
/// never the tighter side.
fn tighter_reach(left: Option<u128>, right: Option<u128>) -> Option<u128> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(only), None) | (None, Some(only)) => Some(only),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sis::decomposition_digits::{
        balanced_digit_abs_max, balanced_digit_max, num_digits_inner_for_bound,
    };
    use crate::sis::honest_fold_policy::{
        BalancedSignedDigitFoldPolicy, UnitOneHotFoldPolicy, DEFAULT_UNIT_ONEHOT_SOURCE_CHUNK_SIZE,
    };

    fn decomposition(log_commit_bound: u32, log_open_bound: Option<u32>) -> DecompositionParams {
        DecompositionParams {
            log_basis: 3,
            log_commit_bound,
            log_open_bound,
        }
    }

    fn balanced(log_commit_bound: u32) -> CommittedSourceContract {
        CommittedSourceContract::try_new(
            CommittedSourceClass::BalancedSignedDigit,
            decomposition(log_commit_bound, Some(128)),
        )
        .expect("valid balanced-digit contract")
    }

    fn one_hot(log_commit_bound: u32) -> CommittedSourceContract {
        CommittedSourceContract::try_new(
            CommittedSourceClass::UnitOneHot {
                source_chunk_size: DEFAULT_UNIT_ONEHOT_SOURCE_CHUNK_SIZE,
            },
            decomposition(log_commit_bound, Some(128)),
        )
        .expect("valid one-hot contract")
    }

    /// The policy comes from the declared class, not from the numeric bound.
    ///
    /// This is the property the bounded-source work exists to preserve: a preset
    /// declares class and bound independently, so neither may be inferred from
    /// the other.
    #[test]
    fn the_policy_is_derived_from_the_class_not_the_bound() {
        assert_eq!(
            CommittedSourceClass::BalancedSignedDigit.honest_fold_policy(128),
            HonestFoldPolicySpec::BalancedSignedDigit(BalancedSignedDigitFoldPolicy::universal(
                128
            ))
        );
        assert_eq!(
            CommittedSourceClass::UnitOneHot {
                source_chunk_size: 256
            }
            .honest_fold_policy(128),
            HonestFoldPolicySpec::UnitOneHot(UnitOneHotFoldPolicy::canonical(128, 256))
        );

        // A balanced-digit source declared at bound 1 stays balanced-digit, and a
        // one-hot source declared at a wide bound stays one-hot. Inferring either
        // from `log_commit_bound` would get both of these wrong.
        assert_eq!(balanced(1).class, CommittedSourceClass::BalancedSignedDigit);
        assert!(matches!(
            one_hot(64).class,
            CommittedSourceClass::UnitOneHot { .. }
        ));
    }

    /// Only the one-hot class imposes a structural requirement.
    #[test]
    fn source_classes_determine_structure_and_decomposed_norms() {
        assert_eq!(
            one_hot(1).class.required_onehot_chunk_size(),
            Some(DEFAULT_UNIT_ONEHOT_SOURCE_CHUNK_SIZE)
        );
        // A one-hot source under a dense schedule is admissible: its digit energy
        // is below what the balanced-digit model charges, so pricing stays
        // conservative.
        assert_eq!(balanced(65).class.required_onehot_chunk_size(), None);
        let small = balanced(2).source_norms(3, 1, 128, 128).unwrap();
        assert_eq!(
            (
                small.fold_witness.infinity_norm(),
                small.fold_witness.l1_norm()
            ),
            (2, 256)
        );
        assert_eq!((small.l2_sq, small.coefficient_sq_max), (512, 4));
        let signed64 = balanced(64).source_norms(5, 13, 128, 1).unwrap();
        assert_eq!(signed64.fold_witness.infinity_norm(), 16);
        assert_eq!(signed64.l2_sq, 12 * 256 + 64);
    }

    /// The declared interval constrains only a bounded balanced-digit source.
    #[test]
    fn declared_bounds_constrain_only_a_bounded_balanced_digit_source() {
        // Full-field endpoint: every field element is in range by construction.
        assert_eq!(balanced(128).declared_bounds(), (None, None));
        assert_eq!(
            CommittedSourceContract::try_new(
                CommittedSourceClass::BalancedSignedDigit,
                decomposition(128, None)
            )
            .expect("full-field contract without an explicit open bound")
            .declared_bounds(),
            (None, None)
        );
        // Unit one-hot: keyed on the class. Reading `k = 1` as `[-1, 0]` would
        // reject a hot position, and the one-hot admitted set is structural.
        assert_eq!(one_hot(1).declared_bounds(), (None, None));
        // ...and that holds whatever bound a one-hot preset happens to declare.
        assert_eq!(one_hot(64).declared_bounds(), (None, None));

        // Interior balanced-digit: the signed reading.
        assert_eq!(
            balanced(64).declared_bounds(),
            (Some(1 << 63), Some((1u128 << 63) - 1))
        );
        // A `u64` workload declares 65: its magnitude reaches `2^64 - 1`.
        assert_eq!(
            balanced(65).declared_bounds(),
            (Some(1 << 64), Some(u128::from(u64::MAX)))
        );
    }

    /// The accepted interval intersects representability with the declaration,
    /// and the declaration binds at every shipped bounded geometry.
    #[test]
    fn accepted_bounds_intersect_representability_with_the_declaration() {
        for (log_basis_inner, num_digits_inner) in [(5u32, 14usize), (9, 8), (7, 10)] {
            let accepted = balanced(65).accepted_bounds(log_basis_inner, num_digits_inner);
            assert_eq!(
                accepted,
                (Some(1 << 64), Some(u128::from(u64::MAX))),
                "the declaration must bind at lb={log_basis_inner} delta={num_digits_inner}"
            );
        }

        // Representability binds when the digits cannot reach the declaration.
        assert_eq!(
            balanced(65).accepted_bounds(5, 2),
            (
                Some(balanced_digit_abs_max(5, 2)),
                Some(balanced_digit_max(5, 2))
            )
        );

        // Full field: only representability, and 12 base-2^11 digits span 132
        // bits, beyond every `u128` on both sides.
        assert_eq!(balanced(128).accepted_bounds(11, 12), (None, None));

        // One-hot: the magnitude side must still admit the hot value `1`.
        let (_, one_hot_positive) = one_hot(1).accepted_bounds(3, 1);
        assert!(one_hot_positive.is_some_and(|reach| reach >= 1));
    }

    /// A contract cannot be built from a decomposition its own methods assume
    /// away.
    ///
    /// `declared_bounds` computes `log_commit_bound - 1`. On a zero bound that
    /// panics in debug and wraps in release, where `checked_shl(u32::MAX)` returns
    /// `None` and the invalid declaration reads as *unconstrained* — the most
    /// permissive answer possible, and a behavior difference between build
    /// profiles. Construction rejects it instead, so no profile can reach that
    /// state.
    #[test]
    fn construction_rejects_a_decomposition_its_methods_assume_away() {
        let invalid = [
            // Zero bound: the subtraction `declared_bounds` performs.
            decomposition(0, Some(128)),
            // Bound above the declared field width.
            decomposition(129, Some(128)),
            // A source bound wider than the field it lives in.
            decomposition(128, Some(64)),
        ];
        for params in invalid {
            assert!(
                matches!(
                    CommittedSourceContract::try_new(
                        CommittedSourceClass::BalancedSignedDigit,
                        params
                    ),
                    Err(AkitaError::InvalidSetup(_))
                ),
                "must reject {params:?}"
            );
        }
        // A degenerate basis is rejected by the same validator.
        assert!(CommittedSourceContract::try_new(
            CommittedSourceClass::BalancedSignedDigit,
            DecompositionParams {
                log_basis: 0,
                log_commit_bound: 64,
                log_open_bound: Some(128),
            },
        )
        .is_err());

        // Class-specific data is validated too: a one-hot chunk size has to be a
        // nonzero power of two for the per-chunk sparsity model to mean anything.
        for chunk in [0usize, 3, 100] {
            assert!(
                CommittedSourceContract::try_new(
                    CommittedSourceClass::UnitOneHot {
                        source_chunk_size: chunk
                    },
                    decomposition(1, Some(128)),
                )
                .is_err(),
                "must reject one-hot chunk size {chunk}"
            );
        }
    }

    /// The gap the intersection closes, in the numbers that motivated it.
    #[test]
    fn representable_envelope_overshoots_the_declaration_it_was_sized_from() {
        for (log_basis_inner, expected_ratio) in [(5u32, 1), (9, 255), (7, 63)] {
            let contract = balanced(64);
            let num_digits_inner = num_digits_inner_for_bound(
                DecompositionParams {
                    log_basis: log_basis_inner,
                    ..contract.decomposition()
                },
                contract.decomposition().log_commit_bound,
            );
            let (_, representable) =
                checked_balanced_digit_representable_bounds(log_basis_inner, num_digits_inner);
            let representable = representable.expect("shipped geometries fit u128");
            let declared = contract.declared_bounds().1.expect("interior bound");
            assert!(
                representable / declared >= expected_ratio,
                "lb={log_basis_inner}: envelope {representable} vs declaration {declared}"
            );
            assert_eq!(
                contract
                    .accepted_bounds(log_basis_inner, num_digits_inner)
                    .1,
                Some(declared)
            );
        }
    }
}
