use super::*;
use crate::commitment::{CommitmentStatePolicy, InnerRelationState, OuterCompressionState};
use crate::compute::{
    ComputeBackendSetup, DigitRowsComputeBackend, LevelProveStacks, RuntimeRingSwitchProveBackend,
};
use jolt_field::AdditiveGroup;

impl<'stack, Stacks: ?Sized> ProverExecutor<'stack, Stacks> {
    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    pub(crate) fn prove_root_native<F, E, P, S, O, TS, R, SP, Cfg>(
        &self,
        expanded: &Arc<AkitaExpandedSetup<F>>,
        prefix_slots: &SetupPrefixProverRegistry<F>,
        grinding: &mut akita_types::NativeProverGrinding<'_>,
        claims: ProverOpeningData<'_, E, P, F, S>,
        commitment_material: Vec<crate::types::PreparedCommitmentRelationMaterial<F>>,
        scheduled: &akita_types::FoldParams,
        next_params: super::fold::FoldSuccessorParams<'_>,
        next_witness_binding: akita_types::NextWitnessBindingPolicy,
        basis: BasisMode,
    ) -> Result<NativeProveLevelOutput<F, E, SP::State>, AkitaError>
    where
        F: Field
            + CanonicalEncoding
            + akita_serialization::AkitaSerialize
            + Unreduced
            + PseudoMersenne
            + Ring
            + 'static,
        <F as Unreduced>::Wide: From<F> + AdditiveGroup,
        E: FpExtEncoding<F>
            + ExtField<F>
            + Unreduced
            + Fold
            + Ring
            + MulBaseUnreduced<F>
            + AkitaSerialize,
        P: RootProverGroupOpening<F, E, O> + Clone,
        S: InnerRelationState<F> + OuterCompressionState<F>,
        O: DigitRowsComputeBackend<F> + ComputeBackendSetup<F> + 'stack,
        TS: ComputeBackendSetup<F> + 'stack,
        R: RuntimeRingSwitchProveBackend<F>
            + DigitRowsComputeBackend<F>
            + ComputeBackendSetup<F>
            + 'stack,
        Cfg: CommitmentConfig<Field = F, ExtField = E>,
        SP: CommitmentStatePolicy<F>,
        SP::State: InnerRelationState<F> + OuterCompressionState<F>,
        <O as ComputeBackendSetup<F>>::PreparedSetup: 'stack,
        <TS as ComputeBackendSetup<F>>::PreparedSetup: 'stack,
        <R as ComputeBackendSetup<F>>::PreparedSetup: 'stack,
        Stacks: LevelProveStacks<
            'stack,
            F,
            Opening = O,
            Tensor = TS,
            RingSwitch = R,
            CommitmentStatePolicy = SP,
        >,
    {
        let stack = self.stacks.prove_stack_at_level(0);
        let root_params = &scheduled.params;
        claims.append_to_native(root_params, grinding)?;
        let prepared_fold = prepare_single_field_fold_native::<F, E, P, S, O, TS, R, SP>(
            stack,
            claims,
            commitment_material,
            false,
            grinding,
            0,
            root_params,
            basis,
        )?;
        prove_fold_native::<F, E, O, TS, R, SP, Cfg>(
            expanded,
            prefix_slots,
            stack,
            grinding,
            0,
            root_params,
            next_params,
            scheduled.output_witness_len,
            next_witness_binding,
            prepared_fold,
        )
    }
}
