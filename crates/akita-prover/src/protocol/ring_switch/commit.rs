use super::*;
use crate::commitment::{CommitmentExecutionPlan, CommitmentExecutor, CommitmentStatePolicy};
use akita_types::{dispatch_for_field, CommittedSourceEncoding, TerminalFoldParams};

/// Public state bound for the witness produced by one intermediate fold.
pub enum NextWitnessState<F: Field> {
    /// Ordinary recursive edge, bound by the terminal compressed payload.
    OuterPayload(RingVec<F>),
    /// Last recursive edge, bound directly by the canonical inner `t` state.
    TerminalInnerState,
}

/// Result of preparing the next logical recursive witness and its public state.
pub struct NextWitnessStateOutput<F: Field, S> {
    /// Physical witness representation when extension packing changes the logical witness.
    pub witness: Option<RecursiveWitnessFlat>,
    /// Transcript-bound public state for the next level.
    pub binding: NextWitnessState<F>,
    /// Prover hint for opening the physical next-level witness.
    pub prover_state: S,
}

/// Commit the next recursive witness under config `Cfg`.
///
/// The commitment ring dimension is schedule-owned (`commit_params.ring_dimension`).
/// This function warms the target NTT slot on the caller's D-free prepared setup,
/// dispatches locally to the typed commit kernel, and returns D-free protocol
/// storage.
///
/// # Errors
///
/// Returns an error if layout selection, commitment, cache preparation, or
/// D-erased hint construction fails.
#[inline(never)]
pub fn commit_w<Cfg, SP>(
    commit_params: &CommittedGroupParams,
    fold_level: usize,
    executor: &CommitmentExecutor<'_, Cfg::Field, SP>,
    logical_w: &RecursiveWitnessFlat,
) -> Result<NextWitnessStateOutput<Cfg::Field, SP::State>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    SP: CommitmentStatePolicy<Cfg::Field>,
{
    let dims = commit_params.role_dims();
    let packed_witness = dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        Cfg::Field,
        dims.d_a(),
        |D_A| {
            let packed_witness = match commit_params.source_encoding {
                CommittedSourceEncoding::CanonicalCoefficientTable => None,
                CommittedSourceEncoding::TensorSubfieldProjection { extension_degree } => {
                    if extension_degree != <Cfg::ExtField as ExtField<Cfg::Field>>::DEGREE {
                        return Err(AkitaError::InvalidSetup(
                            "recursive tensor source encoding does not match the protocol extension degree"
                                .into(),
                        ));
                    }
                    Some(tensor_pack_recursive_witness::<
                        Cfg::Field,
                        Cfg::ExtField,
                        D_A,
                    >(logical_w)?)
                }
            };
            Ok::<_, AkitaError>(packed_witness)
        }
    )?;
    let witness = packed_witness.as_ref().unwrap_or(logical_w);
    let plan = CommitmentExecutionPlan::for_recursive(commit_params, fold_level, 1)?;
    let sources: [&dyn crate::commitment::CommitmentSource<Cfg::Field>; 1] = [witness];
    let (commitment, prover_state) = if commit_params.payload_mode.is_compressed() {
        executor.execute_full(&plan, &sources)?.into_parts()
    } else {
        executor.execute_uncompressed(&plan, &sources)?.into_parts()
    };
    Ok(NextWitnessStateOutput {
        witness: packed_witness,
        binding: NextWitnessState::OuterPayload(commitment),
        prover_state,
    })
}

/// Bind the witness entering the terminal fold with its canonical inner
/// commitment state. No outer digits or outer commitment are computed.
#[inline(never)]
pub fn commit_terminal_w<Cfg, SP>(
    commit_params: &TerminalFoldParams,
    executor: &CommitmentExecutor<'_, Cfg::Field, SP>,
    logical_w: &RecursiveWitnessFlat,
) -> Result<NextWitnessStateOutput<Cfg::Field, SP::State>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    SP: CommitmentStatePolicy<Cfg::Field>,
{
    let ring_dim = commit_params.d_a();
    let packed_witness = dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        Cfg::Field,
        ring_dim,
        |D_A| {
            let packed_witness = if <Cfg::ExtField as ExtField<Cfg::Field>>::DEGREE == 1 {
                None
            } else {
                Some(tensor_pack_recursive_witness::<
                    Cfg::Field,
                    Cfg::ExtField,
                    D_A,
                >(logical_w)?)
            };
            Ok::<_, AkitaError>(packed_witness)
        }
    )?;
    let witness = packed_witness.as_ref().unwrap_or(logical_w);
    let plan = CommitmentExecutionPlan::for_terminal(commit_params)?;
    let sources: [&dyn crate::commitment::CommitmentSource<Cfg::Field>; 1] = [witness];
    let prover_state = executor.execute_inner(&plan, &sources)?.into_state();
    Ok(NextWitnessStateOutput {
        witness: packed_witness,
        binding: NextWitnessState::TerminalInnerState,
        prover_state,
    })
}
