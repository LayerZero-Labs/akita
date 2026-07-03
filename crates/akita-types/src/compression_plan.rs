//! Schedule-side commitment compression plan shapes.

use crate::descriptor_bytes::push_usize;

/// Compression map role for a scalar commitment-compression layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompressionMapRole {
    /// `H_i` map on the D-side opening commitment `v`.
    H,
    /// `F_i` map on a recursive B-side next-witness commitment `u`.
    F,
    /// `F_i` map on the root B-side user commitment.
    RootF,
}

impl CompressionMapRole {
    pub(crate) fn descriptor_tag(self) -> u8 {
        match self {
            Self::H => 0,
            Self::F => 1,
            Self::RootF => 2,
        }
    }
}

/// One scalar compression map layer.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CompressionLayerPlan {
    /// Logical role of the matrix view.
    pub role: CompressionMapRole,
    /// Layer index inside this role, e.g. `0` for `H0`.
    pub layer: usize,
    /// Input digit length in scalar field elements.
    pub input_len: usize,
    /// Output length in scalar field elements.
    pub output_len: usize,
    /// Offset into the shared scalar compression setup prefix.
    pub setup_offset: usize,
}

impl CompressionLayerPlan {
    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        bytes.push(self.role.descriptor_tag());
        push_usize(bytes, self.layer);
        push_usize(bytes, self.input_len);
        push_usize(bytes, self.output_len);
        push_usize(bytes, self.setup_offset);
    }
}

/// Compression plan for one public commitment payload.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CommitmentCompressionPlan {
    /// Raw uncompressed public payload length in scalar field elements.
    pub raw_len: usize,
    /// Final compressed public payload length in scalar field elements.
    pub public_len: usize,
    /// Logical hidden suffix length needed by this commitment.
    pub suffix_len: usize,
    /// Physical hidden suffix length after padding.
    pub padded_suffix_len: usize,
    /// Active scalar compression map layers.
    pub layers: Vec<CompressionLayerPlan>,
}

impl CommitmentCompressionPlan {
    /// Return whether this plan has no active compression.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        push_usize(bytes, self.raw_len);
        push_usize(bytes, self.public_len);
        push_usize(bytes, self.suffix_len);
        push_usize(bytes, self.padded_suffix_len);
        push_usize(bytes, self.layers.len());
        for layer in &self.layers {
            layer.append_descriptor_bytes(bytes);
        }
    }
}

/// Fold-local compression plan.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct FoldCompressionPlan {
    /// D-side `v` compression for this fold.
    pub v: Option<CommitmentCompressionPlan>,
    /// B-side next-witness `u` compression for this fold. `None` on the
    /// penultimate fold in PR2.
    pub next_u: Option<CommitmentCompressionPlan>,
}

impl FoldCompressionPlan {
    /// Return true when neither `v` nor next `u` is compressed.
    pub fn is_empty(&self) -> bool {
        self.v.is_none() && self.next_u.is_none()
    }

    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        append_optional_compression_plan(bytes, self.v.as_ref());
        append_optional_compression_plan(bytes, self.next_u.as_ref());
    }
}

pub(crate) fn append_optional_compression_plan(
    bytes: &mut Vec<u8>,
    plan: Option<&CommitmentCompressionPlan>,
) {
    match plan {
        Some(plan) => {
            bytes.push(1);
            plan.append_descriptor_bytes(bytes);
        }
        None => bytes.push(0),
    }
}

/// Scalar field-element length of an uncompressed public commitment payload.
#[must_use]
pub fn uncompressed_commitment_public_len(params: &crate::LevelParams) -> usize {
    params.b_key.row_len().saturating_mul(params.ring_dimension)
}

/// Scalar field-element length of the public commitment payload implied by
/// `params` and an optional compression plan.
#[must_use]
pub fn scheduled_commitment_public_len(
    params: &crate::LevelParams,
    compression: Option<&CommitmentCompressionPlan>,
) -> usize {
    compression
        .map(|plan| plan.public_len)
        .unwrap_or_else(|| uncompressed_commitment_public_len(params))
}
