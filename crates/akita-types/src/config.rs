//! Shared commitment-config data shapes.

/// Parameters controlling the gadget decomposition depth (called delta in the paper).
///
/// The gadget base is `b = 2^log_basis`. Each ring coefficient with centered
/// magnitude fitting in `log_commit_bound` bits is decomposed into
/// `ceil(log_commit_bound / log_basis)` balanced digits in `[-b/2, b/2)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DecompositionParams {
    /// Base-2 logarithm of the gadget base (e.g. 3 for base-8 digits in [-4, 3]).
    pub log_basis: u32,

    /// Bit-width of the largest coefficient that the commitment decomposition
    /// must represent.
    ///
    /// The centered representation maps each coefficient `c in [0, q)` to the
    /// signed value in `(-q/2, q/2]`. A value of `k` means the signed magnitude
    /// fits in `k` bits, i.e. lies in `[-2^(k-1), 2^(k-1) - 1]`.
    pub log_commit_bound: u32,

    /// Bit-width of the largest coefficient that the opening decomposition
    /// must represent. When `None`, this defaults to `log_commit_bound`.
    pub log_open_bound: Option<u32>,
}

impl DecompositionParams {
    /// Effective field-element bit-width used for opening witnesses.
    pub fn field_bits(self) -> u32 {
        self.log_open_bound.unwrap_or(self.log_commit_bound)
    }
}

/// Verifier strategy for the public setup contribution in the ring-switch row
/// evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetupContributionMode {
    /// Evaluate the setup contribution directly from the expanded setup matrix.
    Direct,
    /// Use the recursive setup-contribution path.
    Recursive,
}
