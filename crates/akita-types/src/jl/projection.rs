//! Checked, schedule-owned geometry for iterated JL projection reductions.

use akita_algebra::jl::{
    TernaryProjectionMatrix, TernaryProjectionShape, MAX_MATERIALIZED_JL_BYTES,
};
use akita_challenges::{derive_balanced_ternary_matrix_seed, expand_balanced_ternary_matrix};
use akita_error::{checked, AkitaError};
use akita_serialization::DEFAULT_MAX_SEQUENCE_LEN;

/// Domain version for the iterated-JL projection protocol.
pub const JL_PROJECTION_PROTOCOL_VERSION: u32 = 1;
/// Only supported matrix-envelope member view: literal upper-left prefix.
pub const JL_PREFIX_POLICY_VERSION: u32 = 1;
/// Defensive maximum number of layers in one projection chain.
pub const MAX_JL_PROJECTION_LAYERS: usize = 64;
/// Defensive maximum number of transcript-derived projection attempts.
pub const MAX_JL_PROJECTION_RETRIES: u32 = 1 << 16;

/// Logical certificate whose source shortness is being reduced.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum JlCertificateId {
    /// Aggregate semantic `Z`.
    ProjZ,
    /// Literal `Ehat || That`.
    ProjEt,
}

impl JlCertificateId {
    const fn tag(self) -> u8 {
        match self {
            Self::ProjZ => 0,
            Self::ProjEt => 1,
        }
    }
}

/// Role-aligned stem or shared tail within a certificate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum JlProjectionStemId {
    /// Aggregate semantic-Z chain.
    Z,
    /// Private literal-E stem.
    E,
    /// Private literal-T stem.
    T,
    /// Shared tail after the aligned E/T selector join.
    EtTail,
}

impl JlProjectionStemId {
    const fn tag(self) -> u8 {
        match self {
            Self::Z => 0,
            Self::E => 1,
            Self::T => 2,
            Self::EtTail => 3,
        }
    }
}

/// Matrix law and reuse policy selected by the public plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum JlMatrixLawId {
    /// Balanced ternary `P(0)=1/2`, `P(+-1)=1/4`, reused as `I_r tensor J`.
    BalancedTernaryRepeatedBlock,
}

impl JlMatrixLawId {
    const fn tag(self) -> u8 {
        match self {
            Self::BalancedTernaryRepeatedBlock => 0,
        }
    }
}

/// Retry-independent domain of one shared matrix envelope.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct JlMatrixEnvelopeDomain {
    schedule_identity: [u8; 32],
    fold_level: u32,
    topological_depth: u16,
    rows: usize,
    cols: usize,
    law: JlMatrixLawId,
}

impl JlMatrixEnvelopeDomain {
    /// Construct one complete retry-independent envelope seed domain.
    pub fn new(
        schedule_identity: [u8; 32],
        fold_level: u32,
        topological_depth: u16,
        rows: usize,
        cols: usize,
        law: JlMatrixLawId,
    ) -> Result<Self, AkitaError> {
        TernaryProjectionShape::new(rows, cols)?;
        Ok(Self {
            schedule_identity,
            fold_level,
            topological_depth,
            rows,
            cols,
            law,
        })
    }

    /// Complete matrix derivation context for one selected retry.
    #[must_use]
    pub const fn with_retry(self, retry_index: u32) -> JlMatrixDerivationContext {
        JlMatrixDerivationContext {
            domain: self,
            retry_index,
        }
    }

    /// Scheduled envelope row count.
    #[must_use]
    pub const fn rows(self) -> usize {
        self.rows
    }

    /// Scheduled envelope column count.
    #[must_use]
    pub const fn cols(self) -> usize {
        self.cols
    }

    /// Topological projection depth shared by every member.
    #[must_use]
    pub const fn topological_depth(self) -> u16 {
        self.topological_depth
    }

    /// Schedule identity authenticating the ordered member manifest.
    #[must_use]
    pub const fn schedule_identity(self) -> [u8; 32] {
        self.schedule_identity
    }

    /// Fold level owning this envelope.
    #[must_use]
    pub const fn fold_level(self) -> u32 {
        self.fold_level
    }
}

/// Complete domain separator for one transcript-derived matrix envelope.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct JlMatrixDerivationContext {
    domain: JlMatrixEnvelopeDomain,
    retry_index: u32,
}

impl JlMatrixDerivationContext {
    /// Canonical domain bytes.
    ///
    /// Certificate, stem, member layer, blocks, and prefix shape are
    /// intentionally absent. They are authenticated by the schedule identity
    /// and ordered member manifest, while every same-depth use shares this
    /// exact envelope.
    pub fn encode(self) -> Result<Vec<u8>, AkitaError> {
        let rows = u64::try_from(self.domain.rows)
            .map_err(|_| AkitaError::InvalidInput("JL row count exceeds u64".into()))?;
        let cols = u64::try_from(self.domain.cols)
            .map_err(|_| AkitaError::InvalidInput("JL column count exceeds u64".into()))?;
        let mut bytes = Vec::with_capacity(
            b"akita/iterated-jl/matrix".len()
                + std::mem::size_of::<u32>() * 4
                + 32
                + std::mem::size_of::<u16>()
                + std::mem::size_of::<u64>() * 2
                + 1,
        );
        bytes.extend_from_slice(b"akita/iterated-jl/matrix");
        bytes.extend_from_slice(&JL_PROJECTION_PROTOCOL_VERSION.to_le_bytes());
        bytes.extend_from_slice(&self.domain.schedule_identity);
        bytes.extend_from_slice(&self.domain.fold_level.to_le_bytes());
        bytes.extend_from_slice(&self.domain.topological_depth.to_le_bytes());
        bytes.extend_from_slice(&rows.to_le_bytes());
        bytes.extend_from_slice(&cols.to_le_bytes());
        bytes.extend_from_slice(&JL_PREFIX_POLICY_VERSION.to_le_bytes());
        bytes.push(self.domain.law.tag());
        bytes.extend_from_slice(&self.retry_index.to_le_bytes());
        Ok(bytes)
    }

    /// Matrix shape committed by this context.
    pub fn shape(self) -> Result<TernaryProjectionShape, AkitaError> {
        TernaryProjectionShape::new(self.domain.rows, self.domain.cols)
    }
}

/// One authenticated upper-left-prefix use in an envelope member manifest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct JlMatrixMember {
    envelope: JlMatrixEnvelopeDomain,
    certificate: JlCertificateId,
    stem: JlProjectionStemId,
    layer: u16,
    rows: usize,
    cols: usize,
}

impl JlMatrixMember {
    /// Construct one scheduled envelope member.
    pub fn new(
        envelope: JlMatrixEnvelopeDomain,
        certificate: JlCertificateId,
        stem: JlProjectionStemId,
        layer: u16,
        rows: usize,
        cols: usize,
    ) -> Result<Self, AkitaError> {
        let stem_matches = matches!(
            (certificate, stem),
            (JlCertificateId::ProjZ, JlProjectionStemId::Z)
                | (
                    JlCertificateId::ProjEt,
                    JlProjectionStemId::E | JlProjectionStemId::T | JlProjectionStemId::EtTail
                )
        );
        if !stem_matches {
            return Err(AkitaError::InvalidInput(
                "JL certificate and projection stem disagree".into(),
            ));
        }
        if rows == 0 || cols == 0 || rows > envelope.rows || cols > envelope.cols {
            return Err(AkitaError::InvalidInput(
                "JL matrix member is not an upper-left envelope prefix".into(),
            ));
        }
        TernaryProjectionShape::new(rows, cols)?;
        Ok(Self {
            envelope,
            certificate,
            stem,
            layer,
            rows,
            cols,
        })
    }

    /// Shared envelope seed domain.
    #[must_use]
    pub const fn envelope(self) -> JlMatrixEnvelopeDomain {
        self.envelope
    }

    /// Member prefix shape.
    pub fn shape(self) -> Result<TernaryProjectionShape, AkitaError> {
        TernaryProjectionShape::new(self.rows, self.cols)
    }

    /// Logical certificate.
    #[must_use]
    pub const fn certificate(self) -> JlCertificateId {
        self.certificate
    }

    /// Role-aligned stem.
    #[must_use]
    pub const fn stem(self) -> JlProjectionStemId {
        self.stem
    }

    /// Layer index within the member's dependency path.
    #[must_use]
    pub const fn layer(self) -> u16 {
        self.layer
    }
}

/// Checked geometry for one `I_r tensor J` layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct JlBlockLayerPlan {
    member: JlMatrixMember,
    blocks: usize,
    input_len: usize,
    output_len: usize,
    block_num_vars: usize,
}

impl JlBlockLayerPlan {
    /// Construct one repeated-block projection layer.
    pub fn new(member: JlMatrixMember, blocks: usize) -> Result<Self, AkitaError> {
        if !blocks.is_power_of_two() {
            return Err(AkitaError::InvalidInput(
                "JL repeated-block layer requires a power-of-two block count".into(),
            ));
        }
        if !member.rows.is_power_of_two() || !member.cols.is_power_of_two() {
            return Err(AkitaError::InvalidInput(
                "JL reduction layer requires power-of-two local rows and columns".into(),
            ));
        }
        let input_len = checked::product([blocks, member.cols])
            .ok_or_else(|| AkitaError::InvalidInput("JL layer input length overflow".into()))?;
        let output_len = checked::product([blocks, member.rows])
            .ok_or_else(|| AkitaError::InvalidInput("JL layer output length overflow".into()))?;
        for (name, len) in [("input", input_len), ("output", output_len)] {
            if len > DEFAULT_MAX_SEQUENCE_LEN {
                return Err(AkitaError::InvalidInput(format!(
                    "JL layer {name} length {len} exceeds proof-sequence bound {DEFAULT_MAX_SEQUENCE_LEN}"
                )));
            }
        }
        let block_num_vars = checked::ceil_log2(blocks)
            .ok_or_else(|| AkitaError::InvalidInput("JL block domain overflow".into()))?;
        Ok(Self {
            member,
            blocks,
            input_len,
            output_len,
            block_num_vars,
        })
    }

    /// Authenticated envelope member used by this layer.
    #[must_use]
    pub const fn matrix_member(self) -> JlMatrixMember {
        self.member
    }

    /// Number of equal-width blocks.
    #[must_use]
    pub const fn blocks(self) -> usize {
        self.blocks
    }

    /// Live input length.
    #[must_use]
    pub const fn input_len(self) -> usize {
        self.input_len
    }

    /// Live output length.
    #[must_use]
    pub const fn output_len(self) -> usize {
        self.output_len
    }

    /// Boolean variables in the block selector axis.
    #[must_use]
    pub const fn block_num_vars(self) -> usize {
        self.block_num_vars
    }

    /// Total sumcheck variables, ordered as column variables then block variables.
    pub fn reduction_num_vars(self) -> Result<usize, AkitaError> {
        checked::sum([self.member.shape()?.col_num_vars()?, self.block_num_vars])
            .ok_or_else(|| AkitaError::InvalidInput("JL reduction dimension overflow".into()))
    }

    /// Total output-point variables, ordered as row then block variables.
    pub fn output_num_vars(self) -> Result<usize, AkitaError> {
        checked::sum([self.member.shape()?.row_num_vars()?, self.block_num_vars])
            .ok_or_else(|| AkitaError::InvalidInput("JL output dimension overflow".into()))
    }
}

/// Checked schedule-derived plan for one forward chain and reverse reduction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JlProjectionChainPlan {
    layers: Vec<JlBlockLayerPlan>,
    final_energy_bound: u128,
}

/// Checked ordered certificate batch and its shared-envelope manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JlProjectionBatchPlan {
    chains: Vec<JlProjectionChainPlan>,
    envelopes: Vec<JlMatrixEnvelopeDomain>,
    max_retries: u32,
}

impl JlProjectionBatchPlan {
    /// Authenticate chain order, member order, and one envelope per depth.
    pub fn new(chains: Vec<JlProjectionChainPlan>, max_retries: u32) -> Result<Self, AkitaError> {
        if chains.is_empty() || chains.len() > DEFAULT_MAX_SEQUENCE_LEN {
            return Err(AkitaError::InvalidInput(
                "JL projection batch chain count is outside the proof-sequence bound".into(),
            ));
        }
        if max_retries == 0 || max_retries > MAX_JL_PROJECTION_RETRIES {
            return Err(AkitaError::InvalidInput(format!(
                "JL retry count must be in 1..={MAX_JL_PROJECTION_RETRIES}"
            )));
        }
        let first_layer = chains
            .first()
            .and_then(|chain| chain.layers.first())
            .copied()
            .ok_or_else(|| AkitaError::InvalidInput("JL projection batch is empty".into()))?;
        let schedule = first_layer.member.envelope.schedule_identity;
        let fold_level = first_layer.member.envelope.fold_level;
        let mut envelopes = Vec::new();
        let mut members = Vec::new();
        let member_count = chains.iter().try_fold(0usize, |count, chain| {
            checked::sum([count, chain.layers.len()])
        });
        let member_count = member_count
            .ok_or_else(|| AkitaError::InvalidInput("JL envelope member count overflow".into()))?;
        if member_count > DEFAULT_MAX_SEQUENCE_LEN {
            return Err(AkitaError::InvalidInput(
                "JL envelope member count exceeds proof-sequence bound".into(),
            ));
        }
        envelopes
            .try_reserve_exact(member_count)
            .map_err(|_| AkitaError::InvalidInput("JL envelope allocation failed".into()))?;
        members
            .try_reserve_exact(member_count)
            .map_err(|_| AkitaError::InvalidInput("JL member-manifest allocation failed".into()))?;
        let mut previous_chain_key = None;
        for chain in &chains {
            let first_member = chain
                .layers
                .first()
                .copied()
                .ok_or_else(|| AkitaError::InvalidInput("JL projection chain is empty".into()))?
                .member;
            if !matches!(
                (first_member.certificate, first_member.stem),
                (JlCertificateId::ProjZ, JlProjectionStemId::Z)
                    | (JlCertificateId::ProjEt, JlProjectionStemId::EtTail)
            ) {
                return Err(AkitaError::InvalidInput(
                    "JL batch foundation accepts Z and already-joined ET-tail sources only".into(),
                ));
            }
            let chain_key = (first_member.certificate.tag(), first_member.stem.tag());
            if previous_chain_key.is_some_and(|previous| previous >= chain_key) {
                return Err(AkitaError::InvalidInput(
                    "JL projection chains are not in canonical certificate/stem order".into(),
                ));
            }
            previous_chain_key = Some(chain_key);
            for layer in &chain.layers {
                let member = layer.member;
                if member.envelope.schedule_identity != schedule
                    || member.envelope.fold_level != fold_level
                {
                    return Err(AkitaError::InvalidInput(
                        "JL projection batch mixes schedule identities or fold levels".into(),
                    ));
                }
                if members.contains(&member) {
                    return Err(AkitaError::InvalidInput(
                        "JL envelope member manifest contains a duplicate use".into(),
                    ));
                }
                members.push(member);
                match envelopes
                    .iter()
                    .position(|domain: &JlMatrixEnvelopeDomain| {
                        domain.topological_depth == member.envelope.topological_depth
                    }) {
                    Some(index) if envelopes.get(index) != Some(&member.envelope) => {
                        return Err(AkitaError::InvalidInput(
                            "JL members at one depth disagree on their shared envelope".into(),
                        ));
                    }
                    Some(_) => {}
                    None => envelopes.push(member.envelope),
                }
            }
        }
        envelopes.sort_by_key(|domain| domain.topological_depth);
        for (index, envelope) in envelopes.iter().enumerate() {
            if usize::from(envelope.topological_depth) != index {
                return Err(AkitaError::InvalidInput(
                    "JL envelope depths must be contiguous from zero".into(),
                ));
            }
            let (max_rows, max_cols) = members
                .iter()
                .filter(|member| member.envelope.topological_depth == envelope.topological_depth)
                .fold((0usize, 0usize), |(rows, cols), member| {
                    (rows.max(member.rows), cols.max(member.cols))
                });
            if envelope.rows != max_rows || envelope.cols != max_cols {
                return Err(AkitaError::InvalidInput(
                    "JL envelope shape is not the exact member-wise maximum at its depth".into(),
                ));
            }
        }
        let mut matrix_bytes = 0usize;
        for envelope in &envelopes {
            matrix_bytes = checked::sum([
                matrix_bytes,
                TernaryProjectionShape::new(envelope.rows, envelope.cols)?.materialized_len(),
            ])
            .ok_or_else(|| {
                AkitaError::InvalidInput("JL batch matrix materialization budget overflow".into())
            })?;
        }
        for member in &members {
            matrix_bytes = checked::sum([matrix_bytes, member.shape()?.materialized_len()])
                .ok_or_else(|| {
                    AkitaError::InvalidInput(
                        "JL batch matrix materialization budget overflow".into(),
                    )
                })?;
        }
        if matrix_bytes > MAX_MATERIALIZED_JL_BYTES {
            return Err(AkitaError::InvalidInput(format!(
                "JL batch matrices require {matrix_bytes} materialized bytes, exceeding the aggregate budget of {MAX_MATERIALIZED_JL_BYTES} bytes"
            )));
        }
        Ok(Self {
            chains,
            envelopes,
            max_retries,
        })
    }

    /// Certificate chains in canonical proof order.
    #[must_use]
    pub fn chains(&self) -> &[JlProjectionChainPlan] {
        &self.chains
    }

    /// Shared matrix envelopes in increasing topological-depth order.
    #[must_use]
    pub fn envelopes(&self) -> &[JlMatrixEnvelopeDomain] {
        &self.envelopes
    }

    /// Validate the one whole-forest candidate carried by the proof.
    pub fn selected_retry(&self, encoded: Option<u32>) -> Result<u32, AkitaError> {
        let retry = match (self.max_retries, encoded) {
            (1, None) => 0,
            (1, Some(_)) => {
                return Err(AkitaError::InvalidInput(
                    "single-attempt JL proof must omit its retry index".into(),
                ));
            }
            (_, Some(retry)) => retry,
            (_, None) => {
                return Err(AkitaError::InvalidInput(
                    "multi-attempt JL proof is missing its retry index".into(),
                ));
            }
        };
        if retry >= self.max_retries {
            return Err(AkitaError::InvalidInput(
                "JL retry index is outside the public batch plan".into(),
            ));
        }
        Ok(retry)
    }

    /// Whether the whole-forest candidate is present on the wire.
    #[must_use]
    pub const fn retry_is_encoded(&self) -> bool {
        self.max_retries > 1
    }

    /// Maximum number of schedule-admitted whole-forest candidates.
    #[must_use]
    pub const fn max_retries(&self) -> u32 {
        self.max_retries
    }

    /// Expand each depth envelope once, in increasing depth order.
    pub fn derive_envelopes(
        &self,
        master_seed: &[u8; 32],
        retry_index: u32,
    ) -> Result<Vec<TernaryProjectionMatrix>, AkitaError> {
        if retry_index >= self.max_retries {
            return Err(AkitaError::InvalidInput(
                "JL retry index is outside the public batch plan".into(),
            ));
        }
        let mut matrices = Vec::new();
        matrices
            .try_reserve_exact(self.envelopes.len())
            .map_err(|_| AkitaError::InvalidInput("JL envelope-list allocation failed".into()))?;
        for envelope in &self.envelopes {
            let context = envelope.with_retry(retry_index);
            let seed = derive_balanced_ternary_matrix_seed(master_seed, &context.encode()?)?;
            matrices.push(expand_balanced_ternary_matrix(&seed, context.shape()?)?);
        }
        Ok(matrices)
    }

    /// Materialize each chain member from already-expanded depth envelopes.
    pub fn member_matrices(
        &self,
        chain_index: usize,
        envelopes: &[TernaryProjectionMatrix],
    ) -> Result<Vec<TernaryProjectionMatrix>, AkitaError> {
        if envelopes.len() != self.envelopes.len() {
            return Err(AkitaError::InvalidSize {
                expected: self.envelopes.len(),
                actual: envelopes.len(),
            });
        }
        let chain = self.chains.get(chain_index).ok_or_else(|| {
            AkitaError::InvalidInput("JL chain index is outside the batch plan".into())
        })?;
        let mut matrices = Vec::new();
        matrices
            .try_reserve_exact(chain.layers.len())
            .map_err(|_| AkitaError::InvalidInput("JL matrix-list allocation failed".into()))?;
        for layer in &chain.layers {
            let depth = usize::from(layer.member.envelope.topological_depth);
            let envelope = envelopes.get(depth).ok_or_else(|| {
                AkitaError::InvalidInput("JL member references a missing depth envelope".into())
            })?;
            let shape = layer.member.shape()?;
            matrices.push(envelope.upper_left_prefix(shape.rows(), shape.cols())?);
        }
        Ok(matrices)
    }
}

impl JlProjectionChainPlan {
    /// Construct a chain. Consecutive live lengths must agree exactly.
    pub fn new(
        layers: Vec<JlBlockLayerPlan>,
        final_energy_bound: u128,
    ) -> Result<Self, AkitaError> {
        if layers.is_empty() || layers.len() > MAX_JL_PROJECTION_LAYERS {
            return Err(AkitaError::InvalidInput(format!(
                "JL chain layer count must be in 1..={MAX_JL_PROJECTION_LAYERS}"
            )));
        }
        for pair in layers.windows(2) {
            if pair[0].output_len != pair[1].input_len {
                return Err(AkitaError::InvalidInput(format!(
                    "JL chain layer boundary mismatch: {} != {}",
                    pair[0].output_len, pair[1].input_len
                )));
            }
        }
        let chain_member = layers
            .first()
            .ok_or_else(|| AkitaError::InvalidInput("JL chain is empty".into()))?
            .member;
        for (index, layer) in layers.iter().enumerate() {
            let expected_layer = u16::try_from(index)
                .map_err(|_| AkitaError::InvalidInput("JL layer index exceeds u16".into()))?;
            if layer.member.layer != expected_layer {
                return Err(AkitaError::InvalidInput(
                    "JL matrix layer tag must equal its forward chain index".into(),
                ));
            }
            if layer.member.envelope.schedule_identity != chain_member.envelope.schedule_identity
                || layer.member.envelope.fold_level != chain_member.envelope.fold_level
                || layer.member.certificate != chain_member.certificate
                || layer.member.stem != chain_member.stem
            {
                return Err(AkitaError::InvalidInput(
                    "JL chain mixes matrix semantic domains".into(),
                ));
            }
            if index > 0
                && layer.member.envelope.topological_depth
                    <= layers[index - 1].member.envelope.topological_depth
            {
                return Err(AkitaError::InvalidInput(
                    "JL dependency path must use strictly increasing envelope depths".into(),
                ));
            }
        }
        Ok(Self {
            layers,
            final_energy_bound,
        })
    }

    /// Layers in forward projection order.
    #[must_use]
    pub fn layers(&self) -> &[JlBlockLayerPlan] {
        &self.layers
    }

    /// Source vector length.
    #[must_use]
    pub fn source_len(&self) -> usize {
        self.layers.first().map_or(0, |layer| layer.input_len)
    }

    /// Clear final image length.
    #[must_use]
    pub fn final_image_len(&self) -> usize {
        self.layers.last().map_or(0, |layer| layer.output_len)
    }

    /// Public accepted final squared energy.
    #[must_use]
    pub const fn final_energy_bound(&self) -> u128 {
        self.final_energy_bound
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn envelope(depth: u16, rows: usize, cols: usize) -> JlMatrixEnvelopeDomain {
        JlMatrixEnvelopeDomain::new(
            [9; 32],
            3,
            depth,
            rows,
            cols,
            JlMatrixLawId::BalancedTernaryRepeatedBlock,
        )
        .unwrap()
    }

    fn member(
        stem: JlProjectionStemId,
        layer: u16,
        depth: u16,
        rows: usize,
        cols: usize,
    ) -> JlMatrixMember {
        let certificate = if stem == JlProjectionStemId::Z {
            JlCertificateId::ProjZ
        } else {
            JlCertificateId::ProjEt
        };
        JlMatrixMember::new(
            envelope(depth, rows.max(4), cols.max(8)),
            certificate,
            stem,
            layer,
            rows,
            cols,
        )
        .unwrap()
    }

    #[test]
    fn envelope_context_binds_seed_coordinates_and_omits_members() {
        let base = envelope(1, 8, 8);
        let encoded = base.with_retry(0).encode().unwrap();
        assert_ne!(encoded, base.with_retry(1).encode().unwrap());
        assert_ne!(encoded, envelope(2, 8, 8).with_retry(0).encode().unwrap());
        let e = JlMatrixMember::new(
            base,
            JlCertificateId::ProjEt,
            JlProjectionStemId::E,
            0,
            4,
            8,
        )
        .unwrap();
        let t = JlMatrixMember::new(
            base,
            JlCertificateId::ProjEt,
            JlProjectionStemId::T,
            0,
            4,
            8,
        )
        .unwrap();
        assert_eq!(e.envelope().with_retry(0).encode().unwrap(), encoded);
        assert_eq!(t.envelope().with_retry(0).encode().unwrap(), encoded);
    }

    #[test]
    fn batch_checks_dependency_order_and_shares_prefix_envelopes() {
        let first = JlBlockLayerPlan::new(member(JlProjectionStemId::Z, 0, 0, 4, 8), 2).unwrap();
        let second = JlBlockLayerPlan::new(member(JlProjectionStemId::Z, 1, 1, 4, 8), 1).unwrap();
        let z = JlProjectionChainPlan::new(vec![first, second], 100).unwrap();
        let et_first =
            JlBlockLayerPlan::new(member(JlProjectionStemId::EtTail, 0, 0, 2, 8), 2).unwrap();
        let et_second =
            JlBlockLayerPlan::new(member(JlProjectionStemId::EtTail, 1, 1, 2, 4), 1).unwrap();
        let et = JlProjectionChainPlan::new(vec![et_first, et_second], 100).unwrap();
        let batch = JlProjectionBatchPlan::new(vec![z, et], 2).unwrap();
        assert_eq!(batch.envelopes().len(), 2);
        let envelopes = batch.derive_envelopes(&[7; 32], 1).unwrap();
        let z_matrices = batch.member_matrices(0, &envelopes).unwrap();
        let et_matrices = batch.member_matrices(1, &envelopes).unwrap();
        for row in 0..2 {
            for col in 0..8 {
                assert_eq!(
                    z_matrices[0].entry(row, col),
                    et_matrices[0].entry(row, col)
                );
            }
        }

        let repeated_depth =
            JlBlockLayerPlan::new(member(JlProjectionStemId::Z, 1, 0, 4, 8), 1).unwrap();
        assert!(JlProjectionChainPlan::new(vec![first, repeated_depth], 100).is_err());
    }

    #[test]
    fn block_layer_rejects_live_tensors_over_proof_sequence_bound() {
        let oversized_input = JlBlockLayerPlan::new(
            member(JlProjectionStemId::Z, 0, 0, 1, 2),
            DEFAULT_MAX_SEQUENCE_LEN,
        );
        assert!(oversized_input.is_err());

        let oversized_output = JlBlockLayerPlan::new(
            member(JlProjectionStemId::Z, 0, 0, 2, 1),
            DEFAULT_MAX_SEQUENCE_LEN,
        );
        assert!(oversized_output.is_err());
    }

    #[test]
    fn whole_forest_retry_is_single_and_bounded() {
        let layer = JlBlockLayerPlan::new(member(JlProjectionStemId::Z, 0, 0, 4, 8), 1).unwrap();
        let chain = JlProjectionChainPlan::new(vec![layer], 100).unwrap();
        let single = JlProjectionBatchPlan::new(vec![chain.clone()], 1).unwrap();
        assert_eq!(single.selected_retry(None).unwrap(), 0);
        assert!(single.selected_retry(Some(0)).is_err());
        let multi = JlProjectionBatchPlan::new(vec![chain], 2).unwrap();
        assert_eq!(multi.selected_retry(Some(1)).unwrap(), 1);
        assert!(multi.selected_retry(None).is_err());
        assert!(multi.selected_retry(Some(2)).is_err());
    }

    #[test]
    fn batch_does_not_claim_unimplemented_private_et_stem_linkage() {
        let e_layer = JlBlockLayerPlan::new(member(JlProjectionStemId::E, 0, 0, 4, 8), 1).unwrap();
        let e_chain = JlProjectionChainPlan::new(vec![e_layer], 100).unwrap();
        assert!(JlProjectionBatchPlan::new(vec![e_chain], 1).is_err());
    }

    #[test]
    fn batch_rejects_aggregate_matrix_materialization_over_budget() {
        let envelope = JlMatrixEnvelopeDomain::new(
            [4; 32],
            0,
            0,
            16_384,
            32_768,
            JlMatrixLawId::BalancedTernaryRepeatedBlock,
        )
        .unwrap();
        let member = JlMatrixMember::new(
            envelope,
            JlCertificateId::ProjZ,
            JlProjectionStemId::Z,
            0,
            16_384,
            32_768,
        )
        .unwrap();
        let layer = JlBlockLayerPlan::new(member, 1).unwrap();
        let chain = JlProjectionChainPlan::new(vec![layer], 100).unwrap();
        assert!(JlProjectionBatchPlan::new(vec![chain], 1).is_err());
    }
}
