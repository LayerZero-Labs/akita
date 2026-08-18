//! Human-readable committed-source annotations for generated schedule tables.
//!
//! A [`CommittedGroupProfile`] and a `CATALOG_IDENTITY` record only the
//! *consequence* of a committed-source class (digit depth at a basis), never the
//! class or the declared bound. These emitters regenerate that context as
//! comments so a reader of `akita-schedules/src/generated/` can see why a bounded
//! family's root rows carry a shallow `num_digits_inner`, and tell a one-hot
//! precommit from a bounded or full-width dense one, without decoding the
//! identity by hand. They are comments only: `CATALOG_IDENTITY.decomposition`
//! stays the single source of truth.

use akita_types::sis::HonestFoldPolicySpec;
use akita_types::DecompositionParams;

/// Widest digit span whose exact accepted interval is still readable in a
/// comment.
///
/// Purely a presentation threshold: past it the interval is a twenty-digit
/// number that tells a reader less than `+/-2^64` does. It is not a correctness
/// boundary — exactness is guaranteed by sourcing both sides from
/// [`akita_types::sis::checked_balanced_digit_representable_bounds`].
const MAX_EXACT_REACH_SPAN_BITS: u128 = 16;

/// One-line description of a frozen precommitted group's committed source.
///
/// A [`akita_types::CommittedGroupProfile`] records geometry and matrices, never
/// the source class or the bound its producer declared — those are offline
/// planning inputs and deliberately absent from runtime schedule identity. What
/// survives is the *consequence*: `num_digits_inner` digits at `log_basis_inner`,
/// which pins the exact coefficient interval the group is binding for.
///
/// Nothing downstream needs the label — a grouped row is keyed on exact
/// descriptor equality, so the wrong producer simply fails to resolve. It exists
/// so a reader of a grouped row can tell a one-hot precommit from a bounded or
/// full-width dense one without recomputing the geometry by hand. `policy` comes
/// from the emit spec's per-key honest fold policies, so the class is the exact
/// one the row was planned against rather than a guess from the digit depth.
///
/// # What this may and may not claim
///
/// The inputs are a descriptor plus the planned class. That is enough to state a
/// **unit one-hot** admitted set exactly — `{0, 1}` with the policy's chunk size —
/// because that source is structurally constrained and the chunk size is part of
/// the class.
///
/// It is **not** enough to state a balanced-digit source's admitted set. That is
/// the producer's `log_commit_bound` intersected with the digit envelope
/// ([`akita_types::sis::CommittedSourceContract::accepted_bounds`]), and a frozen
/// [`akita_types::CommittedGroupProfile`] deliberately records only the
/// consequence of the declaration, never the declaration. `DenseBounded` enforces
/// `[-2^64, 2^64 - 1]` while its 14 base-`2^5` digits span 70 bits, so printing
/// the envelope as an acceptance claim would overstate the admitted set by 32x in
/// a security-sensitive generated artifact.
///
/// So a balanced-digit descriptor names the envelope *as* an envelope, and says
/// the producer's declaration may be tighter.
pub(super) fn precommitted_source_note(
    log_basis: u32,
    digits: usize,
    policy: Option<&HonestFoldPolicySpec>,
) -> String {
    // Total digit span. Exact integer arithmetic on the exponent, so it stays
    // meaningful for the >128-bit spans a full-width row reaches.
    let span_bits = (digits as u128).saturating_mul(u128::from(log_basis));
    let geometry = format!("{digits} x base-2^{log_basis} digits, span {span_bits} bits");
    let description = match policy {
        // Structurally constrained, and the chunk size is part of the class, so
        // the admitted set is exactly stateable.
        Some(HonestFoldPolicySpec::UnitOneHot(one_hot)) => format!(
            "unit one-hot: admits {{0, 1}}, one hot position per {} coefficients; {geometry}",
            one_hot.source_chunk_size()
        ),
        // The producer's declared bound is not recoverable from a descriptor.
        Some(HonestFoldPolicySpec::BalancedSignedDigit(_)) => format!(
            "balanced signed digit: {geometry}, representable envelope {}; the producer's \
             declared log_commit_bound may be tighter",
            representable_envelope(log_basis, digits, span_bits)
        ),
        None => format!(
            "source class unrecorded: {geometry}, representable envelope {}",
            representable_envelope(log_basis, digits, span_bits)
        ),
    };
    format!("                // {description}\n")
}

/// Balanced-digit representable envelope, rendered for a generated comment.
///
/// Exact interval for narrow spans, power-of-two magnitude for wide ones. Both
/// sides come from the *checked* reaches, so exactness is guaranteed rather than
/// incidental: the saturating pair understates past `u128` and would read as a
/// real bound in a generated artifact. `None` on either side means the reach is
/// past `u128`, which forces the magnitude form regardless of width.
fn representable_envelope(log_basis: u32, digits: usize, span_bits: u128) -> String {
    match akita_types::sis::checked_balanced_digit_representable_bounds(log_basis, digits) {
        (Some(negative), Some(positive)) if span_bits <= MAX_EXACT_REACH_SPAN_BITS => {
            format!("[-{negative}, {positive}]")
        }
        _ => format!("about +/-2^{}", span_bits.saturating_sub(1)),
    }
}

/// Banner naming the declared committed-source bound, for bounded families only.
///
/// Emitted for the **interior** of the bound range, `1 < log_commit_bound <
/// field_bits`, and nothing else. The two endpoints are already self-describing
/// and would only be noise:
///
/// - A full-field family decomposes over the whole field width, which is the
///   reader's default assumption.
/// - A unit one-hot family says so in its name, and its source type is
///   *structurally* constrained to `{0, 1}` rather than range-checked, so the
///   "`commit` rejects anything outside it" wording below would misdescribe it.
///
/// An interior bound is the surprising case: the rows carry a deliberately
/// shallow `num_digits_inner`, and without this the only trace of why is a
/// `log_commit_bound` buried in `CATALOG_IDENTITY`.
///
/// This is a comment, not data — `CATALOG_IDENTITY.decomposition` stays the single
/// source of truth, and the banner is regenerated from it on every emit.
pub(super) fn emit_bounded_source_banner(decomposition: DecompositionParams) -> String {
    if !decomposition.has_bounded_committed_source() || decomposition.log_commit_bound <= 1 {
        return String::new();
    }
    let bound = decomposition.log_commit_bound;
    let field_bits = decomposition.field_bits();
    format!(
        "//\n\
         // BOUNDED COMMITTED SOURCE: log_commit_bound = {bound} (field width {field_bits}).\n\
         // Every root row below is sized for a source whose centered coefficients fit\n\
         // {bound} signed bits, i.e. [-2^{}, 2^{} - 1] — not for arbitrary field\n\
         // elements. That is why each root `num_digits_inner` is shallower than a\n\
         // full-width family's. These rows are binding and complete only for\n\
         // polynomials inside that range; `commit` rejects anything outside it.\n\
         // Opening witnesses stay full-width ({field_bits} bits).\n",
        bound - 1,
        bound - 1,
    )
}

#[cfg(test)]
mod tests {
    use super::{emit_bounded_source_banner, precommitted_source_note};
    use akita_types::DecompositionParams;

    fn params(log_commit_bound: u32, log_open_bound: Option<u32>) -> DecompositionParams {
        DecompositionParams {
            log_basis: 3,
            log_commit_bound,
            log_open_bound,
        }
    }

    /// Only the interior of the bound range is banner-worthy.
    ///
    /// The banner exists because a bounded family's shallow `num_digits_inner`
    /// is otherwise unexplained in the generated source. Both endpoints already
    /// explain themselves, and the one-hot endpoint would additionally be
    /// *misdescribed*: its source is structurally `{0, 1}`, not range-checked.
    #[test]
    fn banner_is_emitted_only_for_an_interior_bound() {
        // Full-field endpoints, with and without an explicit open bound.
        assert!(emit_bounded_source_banner(params(128, None)).is_empty());
        assert!(emit_bounded_source_banner(params(128, Some(128))).is_empty());
        // Unit one-hot endpoint.
        assert!(emit_bounded_source_banner(params(1, Some(128))).is_empty());

        for bound in [2u32, 32, 64, 127] {
            assert!(
                !emit_bounded_source_banner(params(bound, Some(128))).is_empty(),
                "bound {bound} is interior and must be announced"
            );
        }
    }

    /// A note may only claim what its inputs can recover.
    ///
    /// The inputs are a descriptor plus the planned class. That determines a unit
    /// one-hot source's admitted set exactly, but **not** a balanced-digit one's:
    /// that is the producer's `log_commit_bound` intersected with the digit
    /// envelope, and a frozen descriptor records only the consequence of the
    /// declaration. So the balanced-digit note names an envelope, never an
    /// acceptance, and the one-hot note names the real `{0, 1}` set rather than
    /// the digit interval that merely contains it.
    #[test]
    fn precommitted_note_claims_only_what_its_inputs_recover() {
        use akita_types::sis::{
            BalancedSignedDigitFoldPolicy, HonestFoldPolicySpec, UnitOneHotFoldPolicy,
        };

        let one_hot = HonestFoldPolicySpec::UnitOneHot(UnitOneHotFoldPolicy::new(128, 1, 256));
        let balanced = HonestFoldPolicySpec::BalancedSignedDigit(
            BalancedSignedDigitFoldPolicy::universal(128),
        );

        // One-hot: the structural set and the chunk size, both exact. It must not
        // claim the `[-4, 3]` digit interval, which merely contains `{0, 1}`.
        let note = precommitted_source_note(3, 1, Some(&one_hot));
        assert!(note.contains("unit one-hot"), "{note}");
        assert!(note.contains("admits {0, 1}"), "{note}");
        assert!(
            note.contains("one hot position per 256 coefficients"),
            "{note}"
        );
        assert!(
            !note.contains("[-4, 3]"),
            "a one-hot source does not admit the digit interval: {note}"
        );

        // Balanced digit: an envelope, explicitly not an acceptance. The shipped
        // `u64` bounded descriptor is 14 base-2^5 digits spanning 70 bits, while
        // its producer enforces `[-2^64, 2^64 - 1]` -- 32x tighter.
        let bounded = precommitted_source_note(5, 14, Some(&balanced));
        assert!(bounded.contains("balanced signed digit"), "{bounded}");
        assert!(bounded.contains("representable envelope"), "{bounded}");
        assert!(bounded.contains("+/-2^69"), "{bounded}");
        assert!(
            bounded.contains("declared log_commit_bound may be tighter"),
            "the envelope must not read as the admitted set: {bounded}"
        );

        // No note may claim acceptance for a balanced-digit descriptor.
        let full = precommitted_source_note(5, 26, Some(&balanced));
        for note in [&bounded, &full] {
            assert!(
                !note.contains("accepts"),
                "a descriptor cannot recover the producer's admitted set: {note}"
            );
        }

        // A saturating reach must never reach a generated file: 26 base-2^5
        // digits overflow `u128`, so the wide form prints an exponent instead.
        let saturated = akita_types::sis::balanced_digit_representable_bounds(5, 26).0;
        assert_eq!(saturated, u128::MAX, "26 base-2^5 digits must saturate");
        assert!(full.contains("span 130 bits"), "{full}");
        assert!(full.contains("+/-2^129"), "{full}");
        assert!(!full.contains(&saturated.to_string()), "{full}");

        // An unlabelled descriptor says so rather than guessing a class.
        assert!(precommitted_source_note(5, 13, None).contains("source class unrecorded"));

        // Every note stays a single comment line.
        for note in [note, bounded, full] {
            assert_eq!(note.lines().count(), 1, "{note}");
            assert!(note.trim_start().starts_with("//"), "{note}");
            assert!(note.ends_with('\n'), "{note}");
        }
    }

    /// The banner states the declared bound and the signed interval it implies.
    #[test]
    fn banner_names_the_bound_and_its_signed_interval() {
        let banner = emit_bounded_source_banner(params(65, Some(128)));
        assert!(banner.contains("log_commit_bound = 65"));
        assert!(banner.contains("field width 128"));
        // The bound is a signed bit width, so 65 spans `[-2^64, 2^64 - 1]` --
        // the smallest declaration containing every `u64`.
        assert!(banner.contains("[-2^64, 2^64 - 1]"));
        // Every line stays a comment: this is prepended to generated Rust source.
        assert!(banner.lines().all(|line| line.starts_with("//")));
        assert!(banner.ends_with('\n'));
    }
}
