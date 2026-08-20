use super::{
    FoldSchedule, FoldScheduleDescriptorStep, GroupCommitPhaseParams, TerminalFoldDescriptor,
    TerminalFoldStep,
};
use crate::descriptor_bytes::push_usize;
use crate::layout::params::append_schedule_sparse_challenge_descriptor_bytes;
use crate::CommittedGroupParams;
use akita_field::AkitaError;

impl FoldSchedule {
    /// Canonical byte encoding used to order semantically distinct schedules.
    ///
    /// This is an ordering descriptor, not a wire encoding or transcript
    /// commitment. It includes every schedule field that can affect proving or
    /// verification.
    pub fn canonical_descriptor_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        self.append_descriptor_bytes(&mut bytes);
        bytes
    }

    /// Encode checked fold parts with the exact canonical schedule field order.
    ///
    /// This ordering helper intentionally performs no schedule validation. The
    /// planner uses it for candidates that were already checked by construction;
    /// the selected schedule still goes through full materialization and
    /// structural validation.
    pub fn append_descriptor_bytes_from_steps<'a>(
        bytes: &mut Vec<u8>,
        mut folds: impl ExactSizeIterator<Item = FoldScheduleDescriptorStep<'a>>,
        terminal: TerminalFoldDescriptor<'_>,
    ) -> Result<(), AkitaError> {
        let root = folds.next().ok_or_else(|| {
            AkitaError::UnsupportedSchedule(
                "a fold schedule descriptor requires a root fold".to_string(),
            )
        })?;
        bytes.push(1);
        append_root_fold_descriptor_bytes(
            bytes,
            root.params,
            root.payload_mode,
            root.params
                .precommitted_groups
                .iter()
                .map(|group| (&group.profile, group)),
            &root.params.open_commit_matrix,
            &root.params.fold_challenge_config,
            WitnessChunkDescriptor(root.params.witness_chunk.num_chunks),
            root.input_witness_len,
            root.output_witness_len,
        );
        push_usize(bytes, folds.len());
        for fold in folds {
            append_recursive_fold_descriptor_bytes(
                bytes,
                fold.params,
                fold.payload_mode,
                &fold.params.open_commit_matrix,
                &fold.params.fold_challenge_config,
                fold.params.setup_prefix.as_ref(),
                WitnessChunkDescriptor(fold.params.witness_chunk.num_chunks),
                fold.input_witness_len,
                fold.output_witness_len,
            );
        }
        terminal.append_descriptor_bytes(bytes);
        Ok(())
    }

    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        bytes.push(1);
        append_root_fold_descriptor_bytes(
            bytes,
            &self.root.params,
            self.root.params.payload_mode,
            self.root
                .params
                .precommitted_groups
                .iter()
                .map(|group| (&group.profile, group)),
            &self.root.params.open_commit_matrix,
            &self.root.params.fold_challenge_config,
            WitnessChunkDescriptor(self.root.params.witness_chunk.num_chunks),
            self.root.input_witness_len,
            self.root.output_witness_len,
        );
        push_usize(bytes, self.recursive_folds.len());
        for fold in &self.recursive_folds {
            append_recursive_fold_descriptor_bytes(
                bytes,
                &fold.params,
                fold.params.payload_mode,
                &fold.params.open_commit_matrix,
                &fold.params.fold_challenge_config,
                fold.params.setup_prefix.as_ref(),
                WitnessChunkDescriptor(fold.params.witness_chunk.num_chunks),
                fold.input_witness_len,
                fold.output_witness_len,
            );
        }
        self.terminal.append_descriptor_bytes(bytes);
    }
}

/// Number of witness chunks this fold declares.
///
/// `1` encodes as a bare `0` tag, matching the historical `WitnessPartition::Single`
/// arm exactly; any other count encodes as `(1, count)`. Both of the old
/// representations agreed on this, which is why deleting the mirror is
/// byte-neutral.
struct WitnessChunkDescriptor(usize);

#[allow(clippy::too_many_arguments)]
fn append_root_fold_descriptor_bytes<'a>(
    bytes: &mut Vec<u8>,
    commitment: &CommittedGroupParams,
    payload_mode: crate::CommitmentPayloadMode,
    precommitted_groups: impl ExactSizeIterator<
        Item = (&'a GroupCommitPhaseParams, &'a crate::GroupOpenPhaseParams),
    >,
    open_commit_matrix: &crate::OpenCommitMatrixParams,
    sparse_challenge_config: &akita_challenges::SparseChallengeConfig,
    witness_partition: WitnessChunkDescriptor,
    input_witness_len: usize,
    output_witness_len: usize,
) {
    commitment.append_descriptor_bytes_with_payload_mode(bytes, payload_mode);
    push_usize(bytes, precommitted_groups.len());
    for (descriptor, commitment) in precommitted_groups {
        descriptor.append_descriptor_bytes(bytes);
        commitment.append_descriptor_bytes(bytes);
    }
    open_commit_matrix.append_descriptor_bytes(bytes);
    append_schedule_sparse_challenge_descriptor_bytes(bytes, sparse_challenge_config);
    append_witness_partition_descriptor(bytes, witness_partition);
    push_usize(bytes, input_witness_len);
    push_usize(bytes, output_witness_len);
}

#[allow(clippy::too_many_arguments)]
fn append_recursive_fold_descriptor_bytes(
    bytes: &mut Vec<u8>,
    witness: &CommittedGroupParams,
    payload_mode: crate::CommitmentPayloadMode,
    open_commit_matrix: &crate::OpenCommitMatrixParams,
    sparse_challenge_config: &akita_challenges::SparseChallengeConfig,
    incoming_setup_prefix: Option<&crate::GroupOpenPhaseParams>,
    witness_partition: WitnessChunkDescriptor,
    input_witness_len: usize,
    output_witness_len: usize,
) {
    witness.append_descriptor_bytes_with_payload_mode(bytes, payload_mode);
    open_commit_matrix.append_descriptor_bytes(bytes);
    append_schedule_sparse_challenge_descriptor_bytes(bytes, sparse_challenge_config);
    match incoming_setup_prefix {
        None => bytes.push(0),
        Some(prefix) => {
            bytes.push(1);
            prefix.append_setup_prefix_descriptor_bytes(bytes);
        }
    }
    append_witness_partition_descriptor(bytes, witness_partition);
    push_usize(bytes, input_witness_len);
    push_usize(bytes, output_witness_len);
}

impl TerminalFoldStep {
    /// Canonical ordering descriptor for a terminal suffix.
    pub fn canonical_descriptor_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        self.append_descriptor_bytes(&mut bytes);
        bytes
    }

    fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        TerminalFoldDescriptor {
            witness: &self.params.witness,
            sparse_challenge_config: &self.params.sparse_challenge_config,
            response_shape: &self.params.response_shape,
            input_witness_len: self.input_witness_len,
        }
        .append_descriptor_bytes(bytes);
    }
}

impl TerminalFoldDescriptor<'_> {
    /// Canonical ordering descriptor for a borrowed terminal suffix.
    pub fn canonical_descriptor_bytes(self) -> Vec<u8> {
        let mut bytes = Vec::new();
        self.append_descriptor_bytes(&mut bytes);
        bytes
    }

    fn append_descriptor_bytes(self, bytes: &mut Vec<u8>) {
        bytes.push(3);
        self.witness.append_descriptor_bytes(bytes);
        append_schedule_sparse_challenge_descriptor_bytes(bytes, self.sparse_challenge_config);
        self.response_shape.append_descriptor_bytes(bytes);
        push_usize(bytes, self.input_witness_len);
    }
}

fn append_witness_partition_descriptor(bytes: &mut Vec<u8>, chunks: WitnessChunkDescriptor) {
    match chunks.0 {
        1 => bytes.push(0),
        num_chunks => {
            bytes.push(1);
            push_usize(bytes, num_chunks);
        }
    }
}
