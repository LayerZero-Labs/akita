//! One relation and one typed prefix for all admitted root sources.

use super::*;
use crate::binary_source::{
    GroupedTypedPrefix, OrdinaryNativeRootClaim, TypedPrefixWeights,
};

pub struct GroupedBinaryRelation<'a, E: Field> {
    pub challenges: &'a Challenges,
    pub commitment: &'a [E],
    pub d_payload: &'a [E],
    pub matrix: Option<crate::binary_source::BinaryAOperand<'a, E>>,
}

pub struct GroupedRootPublic {
    binding: GroupedTypedPrefix,
    binary: Vec<BinaryRootPublic>,
    ranges: Vec<Range<usize>>,
    ordinary_range: Range<usize>,
    second_ranges: Vec<Range<usize>>,
    negative: Vec<usize>,
    negative_range: Range<usize>,
}

impl GroupedRootPublic {
    pub fn new(
        binding: GroupedTypedPrefix,
        terminals: &[Option<BinaryTerminalClaim>],
    ) -> Result<Self, AkitaError> {
        let plan = binding.plan();
        let ordinary = plan.ordinary().ok_or(AkitaError::InvalidProof)?;
        if plan.binary().len() != 4 || ordinary.plans().len() != 4 || terminals.len() != 4 {
            return Err(AkitaError::InvalidProof);
        }
        let mut binary = Vec::new();
        let mut ranges = Vec::new();
        let mut offset = 0;
        for (local, terminal) in plan.binary().iter().zip(terminals) {
            let source = local.source();
            if source.word != terminal.is_some() {
                return Err(AkitaError::InvalidProof);
            }
            let public = BinaryRootPublic::new(
                &source.manifest, terminal.as_ref(), &source.prime, binding.setup(),
                local.local().repacking().domain_len(),
                plan.repacking().destination_ring_dimension(), None,
            )?;
            let range = checked::range(offset, public.batch_count())
                .ok_or(AkitaError::InvalidProof)?;
            offset = range.end;
            ranges.push(range);
            binary.push(public);
        }
        let ordinary_range = checked::range(offset, ordinary.linear_batch_count())
            .ok_or(AkitaError::InvalidProof)?;
        offset = ordinary_range.end;
        let mut second_ranges = Vec::new();
        let mut negative = Vec::new();
        for source in ordinary.plans() {
            let range = checked::range(offset, source.profile().group.num_polynomials())
                .ok_or(AkitaError::InvalidProof)?;
            offset = range.end;
            second_ranges.push(range);
            let addresses = source.negative_binary_addresses(binding.outgoing())?;
            negative.try_reserve_exact(addresses.len()).map_err(|_| AkitaError::InvalidProof)?;
            negative.extend(addresses);
        }
        let negative_range =
            checked::range(offset, negative.len()).ok_or(AkitaError::InvalidProof)?;
        Ok(Self {
            binding, binary, ranges, ordinary_range, second_ranges, negative, negative_range,
        })
    }

    pub fn batch_count(&self) -> usize {
        self.negative_range.end
    }

    pub fn binding(&self) -> &GroupedTypedPrefix {
        &self.binding
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn relation<F: Field, E: ExtField<F>>(
        &self,
        binary: &[GroupedBinaryRelation<'_, E>],
        ordinary: &[OrdinaryNativeRootClaim<'_, E>],
        second: &[OrdinaryPointEvaluation<'_, E>],
        alpha: E,
        batches: &[E],
    ) -> Result<(NativeRootWeights<E>, Vec<(usize, E)>), AkitaError> {
        if binary.len() != self.binary.len() || ordinary.len() != 4 || second.len() != 4
            || batches.len() != self.batch_count()
        {
            return Err(AkitaError::InvalidProof);
        }
        let domain = self.binding.outgoing().domain_len();
        let mut components = Vec::new();
        for (((public, input), range), local) in self.binary.iter().zip(binary)
            .zip(&self.ranges).zip(self.binding.plan().binary())
        {
            let (weights, negative) = public.relation::<F, E>(
                input.challenges,
                alpha,
                batches.get(range.clone()).ok_or(AkitaError::InvalidProof)?,
                &[], &[], input.commitment, input.d_payload, None, input.matrix,
            )?;
            if !negative.is_empty() {
                return Err(AkitaError::InvalidProof);
            }
            components.push(weights.translated(local.coefficient_offset(), domain)?);
        }
        let signed = self.binding.plan().ordinary().ok_or(AkitaError::InvalidProof)?;
        components.push(signed.linear_weights(
            self.binding.outgoing(), self.binding.setup(), ordinary, alpha,
            batches.get(self.ordinary_range.clone()).ok_or(AkitaError::InvalidProof)?,
        )?);
        for (((source, reading), first), range) in signed.plans().iter().zip(second)
            .zip(ordinary).zip(&self.second_ranges)
        {
            if reading.point == first.point {
                return Err(AkitaError::InvalidProof);
            }
            components.push(source.evaluation_weights(
                self.binding.outgoing(), reading.point, reading.evaluations,
                batches.get(range.clone()).ok_or(AkitaError::InvalidProof)?,
            )?);
        }
        let mut negative = Vec::new();
        negative.try_reserve_exact(self.negative.len()).map_err(|_| AkitaError::InvalidProof)?;
        negative.extend(self.negative.iter().copied().zip(
            batches.get(self.negative_range.clone()).ok_or(AkitaError::InvalidProof)?
                .iter().copied(),
        ));
        Ok((NativeRootWeights::merge(domain, components, E::zero())?, negative))
    }

    pub fn stage3_weights<E: Field>(
        &self,
        point: &[E],
        alpha: E,
        batches: &[E],
    ) -> Result<TypedPrefixWeights<E>, AkitaError> {
        if batches.len() != self.batch_count()
            || point.len() != self.binding.outgoing().domain_len().trailing_zeros() as usize
        {
            return Err(AkitaError::InvalidProof);
        }
        let mut parts = Vec::new();
        for (((public, range), local), (prefix, typed)) in self.binary.iter().zip(&self.ranges)
            .zip(self.binding.plan().binary()).zip(self.binding.parts())
        {
            let batches = batches.get(range.clone()).ok_or(AkitaError::InvalidProof)?;
            let (rho, _) = public.coefficient_inputs(batches, &[], &[])?;
            parts.push((prefix.start, typed.translated_weights(
                public.plan(), &rho, point, local.coefficient_offset(), alpha,
                public.authentication_batches(batches)?,
            )?));
        }
        TypedPrefixWeights::from_translated(self.binding.first().n_prefix()?, parts)
    }
}
