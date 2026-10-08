//! Typed successor exchange. Backends and bridges own packet representations.
use super::{OpaqueProverConsumer, RecursiveWitnessManifest, ValidatedSuccessorHandoffPlan};
use akita_error::AkitaError;
use jolt_field::{CanonicalEncoding, Field};
use std::any::TypeId;

/// Native identities are namespaced by implementation, never compared as bare counters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendInstanceIdentity {
    implementation: TypeId,
    instance: u128,
}

impl BackendInstanceIdentity {
    pub fn new<B: 'static>(instance: u128) -> Self {
        Self {
            implementation: TypeId::of::<B>(),
            instance,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SuccessorSection {
    LogicalDigits,
    CommittedDigits,
    InnerRows,
    CompressionDigits(usize),
    CompressionQuotient(usize),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SuccessorEncoding {
    SignedI8,
    PackedSigned { bit_width: u8 },
    CanonicalField,
    NegativeBinary,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SuccessorSectionDescriptor {
    pub section: SuccessorSection,
    pub encoding: SuccessorEncoding,
    pub coefficients: usize,
    pub bytes: usize,
}

/// Shape metadata does not grant access to private sections.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SuccessorExportDescriptor {
    pub handoff: u128,
    pub manifest: RecursiveWitnessManifest,
    pub sections: Vec<SuccessorSectionDescriptor>,
}

/// Import metadata is checked against the admitted handoff before adoption.
/// Concrete backends expose constructors for their own packet representation.
pub trait SuccessorImportPacket: Send + 'static {
    fn descriptor(&self) -> &SuccessorExportDescriptor;
}

/// An importer publishes both destination handles together, after validation.
pub struct ImportedSuccessor<W, M> {
    witness: W,
    material: M,
}

impl<W, M> ImportedSuccessor<W, M> {
    /// Backend implementations must validate scope, phase and pairing before issuance.
    pub fn new(witness: W, material: M) -> Self {
        Self { witness, material }
    }
    pub fn into_parts(self) -> (W, M) {
        (self.witness, self.material)
    }
}

/// Produces an owned packet while retaining the producer's original handles.
pub trait SuccessorExportKernel<F: Field + CanonicalEncoding, E: Field>:
    OpaqueProverConsumer<F, E>
{
    type ExportPacket: Send + 'static;

    fn instance_identity(&self) -> BackendInstanceIdentity;

    fn export_successor(
        &self,
        session: &Self::ProofSessionHandle,
        witness: &Self::WitnessHandle,
        material: &Self::CommitmentMaterialHandle,
        plan: &ValidatedSuccessorHandoffPlan<'_, F>,
    ) -> Result<Self::ExportPacket, AkitaError>;
}

/// Validates a destination packet and publishes a fresh local handle pair.
pub trait SuccessorImportKernel<F: Field + CanonicalEncoding, E: Field>:
    OpaqueProverConsumer<F, E>
{
    type ImportPacket: SuccessorImportPacket;

    fn import_successor(
        &self,
        session: &Self::ProofSessionHandle,
        plan: &ValidatedSuccessorHandoffPlan<'_, F>,
        packet: Self::ImportPacket,
    ) -> Result<ImportedSuccessor<Self::WitnessHandle, Self::CommitmentMaterialHandle>, AkitaError>;
}

/// A directed conversion between backend types, registered independently of instances.
pub struct Edge<A, B>(std::marker::PhantomData<fn(A) -> B>);

impl<A, B> Default for Edge<A, B> {
    fn default() -> Self {
        Self(std::marker::PhantomData)
    }
}

/// Implement on `Edge<A, B>` beside a backend to define its supported conversions.
/// The field parameters identify the kernel instantiation; the registry infers them.
pub trait SuccessorBridge<A, B, F: Field + CanonicalEncoding, E: Field>
where
    A: SuccessorExportKernel<F, E>,
    B: SuccessorImportKernel<F, E>,
{
    fn convert(
        packet: A::ExportPacket,
        plan: &ValidatedSuccessorHandoffPlan<'_, F>,
    ) -> Result<B::ImportPacket, AkitaError>;
}
