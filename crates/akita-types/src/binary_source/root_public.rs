//! Shared public compilation of the binary root's fused linear relation.

use akita_challenges::Challenges;
use akita_error::{checked, AkitaError};
use akita_params::binary_setup::{derive_centered_q_matrix, SetupStreamLabel, SetupStreamRole};
use akita_params::{witness_commitment_domain_len, GroupCommitPhaseParams};
use jolt_field::Field;

use super::{
    binary_digit_authentication_weights, BinaryCommittedGroupManifest, BinaryRootWitnessPlan,
    BinaryTerminalClaim, OrdinaryNativeRootPlan,
};
use crate::native_root_relation::NativeRootWeights;
use crate::AkitaSetupDescriptor;
use std::ops::Range;

/// Materialize the q24 source matrix from its canonical typed setup identity.
///
/// Matrix order is A row, source ring row, then natural coefficient. Both the
/// source commitment and setup preparation use this function; online verification does not.
pub fn binary_root_matrix(
    seed: &[u8; 32],
    manifest: &BinaryCommittedGroupManifest,
) -> Result<Vec<Vec<[i128; 648]>>, AkitaError> {
    #[cfg(feature = "test-support")]
    super::matrix_access::check()?;
    manifest.validate()?;
    let columns = checked::product([4, manifest.descriptor().index_map().columns()])
        .and_then(|block| {
            checked::exact_div(manifest.descriptor().index_map().padded_carriers(), block)
        })
        .ok_or(AkitaError::InvalidProof)?;
    let coefficients = derive_centered_q_matrix(seed, manifest.a_label(), columns)?;
    let row_width = checked::product([columns, 648]).ok_or(AkitaError::InvalidProof)?;
    let mut matrix = Vec::new();
    matrix
        .try_reserve_exact(manifest.a_rank())
        .map_err(|_| AkitaError::InvalidProof)?;
    for row in coefficients.chunks_exact(row_width) {
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(columns)
            .map_err(|_| AkitaError::InvalidProof)?;
        for entry in row.chunks_exact(648) {
            let mut lifted = [0i128; 648];
            for (output, &coefficient) in lifted.iter_mut().zip(entry) {
                *output = i128::from(coefficient);
            }
            entries.push(lifted);
        }
        matrix.push(entries);
    }
    if matrix.len() != manifest.a_rank() {
        return Err(AkitaError::InvalidProof);
    }
    Ok(matrix)
}

/// Admitted public inputs and non-A weights for one binary group.
///
/// This experimental root uses the fixed proof prime 2^128-275. Its outgoing
/// coefficient map is authenticated by the ordinary successor commitment;
/// source image digits, U, all carries, and every virtual zero share that map.
/// No security admission of the experimental source matrices is asserted.
#[derive(Clone, Debug)]
pub struct BinaryRootPublic {
    manifest: BinaryCommittedGroupManifest,
    claim: Option<BinaryTerminalClaim>,
    prime: super::PrimePlaneManifest,
    setup: AkitaSetupDescriptor,
    plan: BinaryRootWitnessPlan,
    ordinary: Option<OrdinaryNativeRootPlan>,
    negative_binary_addresses: Vec<usize>,
    batch_count: usize,
}

impl BinaryRootPublic {
    pub fn new(
        manifest: &BinaryCommittedGroupManifest,
        claim: Option<&BinaryTerminalClaim>,
        prime: &super::PrimePlaneManifest,
        setup: &AkitaSetupDescriptor,
        logical_len: usize,
        destination_ring_dimension: usize,
        ordinary_profile: Option<GroupCommitPhaseParams>,
    ) -> Result<Self, AkitaError> {
        manifest.validate()?;
        if let Some(claim) = claim {
            claim.validate(manifest.descriptor())?;
        }
        super::PrimePlaneManifest::new(manifest.descriptor(), prime.planes())?;
        let domain_len = witness_commitment_domain_len(logical_len, destination_ring_dimension)?;
        let mut plan = BinaryRootWitnessPlan::new(
            manifest.descriptor(),
            manifest.a_rank(),
            destination_ring_dimension,
        )?
        .with_prime_planes(claim.is_some(), prime.planes().len())?;
        let ordinary = ordinary_profile
            .map(|profile| OrdinaryNativeRootPlan::new(profile, plan.repacking().segments().len()))
            .transpose()?;
        if let Some(ordinary) = &ordinary {
            plan = plan.with_ordinary(ordinary)?;
        }
        let plan = plan.with_domain_len(domain_len)?;
        let negative_binary_addresses = ordinary
            .as_ref()
            .map(|ordinary| ordinary.negative_binary_addresses(plan.repacking()))
            .transpose()?
            .unwrap_or_default();
        // Every root row reads an address below the natural end, so this one
        // comparison keeps all of them inside the committed witness.
        if plan.repacking().natural_end()? > logical_len {
            return Err(AkitaError::InvalidSetup(
                "binary root natural digits exceed the outgoing witness".into(),
            ));
        }
        let batch_count = checked::product([2, manifest.b_rows()])
            .and_then(|authentication| {
                checked::sum([
                    7,
                    if claim.is_some() { 2 } else { 0 },
                    checked::product([2, prime.planes().len()])?,
                    authentication,
                    ordinary
                        .as_ref()
                        .map_or(0, OrdinaryNativeRootPlan::linear_batch_count),
                    negative_binary_addresses.len(),
                ])
            })
            .ok_or(AkitaError::InvalidProof)?;
        Ok(Self {
            manifest: manifest.clone(),
            claim: claim.cloned(),
            prime: prime.clone(),
            setup: setup.clone(),
            plan,
            ordinary,
            negative_binary_addresses,
            batch_count,
        })
    }

    pub fn plan(&self) -> &BinaryRootWitnessPlan {
        &self.plan
    }

    /// Seven projection outputs, optional word parity/value, B/D, ordinary
    /// rows, negative bits, then two batches for every prime plane. No row
    /// reads a padding address, and none constrains one.
    pub fn batch_count(&self) -> usize {
        self.batch_count
    }

    /// Batch ranges: terminal, B, D, ordinary rows, negative bits, prime rows.
    fn batch_ranges(&self) -> Result<[Range<usize>; 6], AkitaError> {
        let terminal = checked::range(7, if self.claim.is_some() { 2 } else { 0 })
            .ok_or(AkitaError::InvalidProof)?;
        let b =
            checked::range(terminal.end, self.manifest.b_rows()).ok_or(AkitaError::InvalidProof)?;
        let d = checked::range(b.end, self.manifest.b_rows()).ok_or(AkitaError::InvalidProof)?;
        let ordinary_rows = checked::range(
            d.end,
            self.ordinary
                .as_ref()
                .map_or(0, OrdinaryNativeRootPlan::linear_batch_count),
        )
        .ok_or(AkitaError::InvalidProof)?;
        let negative_bits = checked::range(ordinary_rows.end, self.negative_binary_addresses.len())
            .ok_or(AkitaError::InvalidProof)?;
        let prime_rows = checked::range(
            negative_bits.end,
            checked::product([2, self.prime.planes().len()]).ok_or(AkitaError::InvalidProof)?,
        )
        .ok_or(AkitaError::InvalidProof)?;
        Ok([terminal, b, d, ordinary_rows, negative_bits, prime_rows])
    }

    /// Row batches of the B and D authentication rows, in that order.
    pub fn authentication_batches<'a, E>(
        &self,
        batches: &'a [E],
    ) -> Result<[&'a [E]; 2], AkitaError> {
        if batches.len() != self.batch_count {
            return Err(AkitaError::InvalidProof);
        }
        let [_, b, d, ..] = self.batch_ranges()?;
        Ok([
            batches.get(b).ok_or(AkitaError::InvalidProof)?,
            batches.get(d).ok_or(AkitaError::InvalidProof)?,
        ])
    }

    /// Compile the single linear oracle after the outgoing commitment.
    ///
    /// Batch coordinates are independent field challenges, one per row.
    /// `commitment` is the public B payload; `d_payload`
    /// is the order-three commitment to the selected word and prime views.
    /// Proving supplies A; verification omits A and the B/D weights and
    /// authenticates their contribution through the typed prefix. `rho` batches the public projection rows with mu.
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    pub fn relation<E: Field>(
        &self,
        challenges: &Challenges,
        alpha: E,
        batches: &[E],
        rho: &[[E; 648]],
        projected_values: &[i128; 7],
        commitment: &[E],
        d_payload: &[E],
        ordinary_claim: Option<(&[E], E, &[E])>,
        matrix: Option<&[Vec<[i128; 648]>]>,
    ) -> Result<(NativeRootWeights<E>, Vec<(usize, E)>), AkitaError> {
        if batches.len() != self.batch_count {
            return Err(AkitaError::InvalidSize {
                expected: self.batch_count,
                actual: batches.len(),
            });
        }
        let mu: [E; 7] = batches
            .get(..7)
            .ok_or(AkitaError::InvalidProof)?
            .try_into()
            .map_err(|_| AkitaError::InvalidProof)?;
        let projection_claim = mu
            .iter()
            .zip(projected_values)
            .fold(E::zero(), |sum, (&mu, &y)| sum + mu * E::from_i128(y));
        let [terminal, b, d, ordinary_rows, negative_bits, prime_rows] = self.batch_ranges()?;
        let span = tracing::info_span!("binary_root_weights_algebra").entered();
        let algebra = self.plan.relation_weights(
            self.claim.as_ref(),
            matrix,
            challenges,
            alpha,
            rho,
            batches.get(terminal).ok_or(AkitaError::InvalidProof)?,
        )?;
        drop(span);
        let span = tracing::info_span!("binary_root_weights_prime").entered();
        let prime_weights = if let Some(segment) = self.plan.prime_segment() {
            Some(super::prime_plane_relation_weights(
                self.manifest.descriptor(),
                &self.prime,
                self.plan.repacking(),
                0,
                segment,
                challenges,
                alpha,
                batches.get(prime_rows).ok_or(AkitaError::InvalidProof)?,
            )?)
        } else {
            None
        };
        drop(span);
        let span = tracing::info_span!("binary_root_weights_b").entered();
        let b_weights = binary_digit_authentication_weights(
            &self.plan,
            &[1],
            &self.setup.setup_seed.seed,
            self.manifest.b_label(),
            commitment,
            alpha,
            batches.get(b).ok_or(AkitaError::InvalidProof)?,
        )?;
        drop(span);
        let span = tracing::info_span!("binary_root_weights_d").entered();
        let d_weights = binary_digit_authentication_weights(
            &self.plan,
            &self.plan.opening_segments(),
            &self.setup.setup_seed.seed,
            SetupStreamLabel {
                role: SetupStreamRole::WitnessCommitment,
                ..self.manifest.b_label()
            },
            d_payload,
            alpha,
            batches.get(d).ok_or(AkitaError::InvalidProof)?,
        )?;
        drop(span);
        // The typed prefix carries the B/D weights into Stage 3; verification
        // keeps only their claims.
        let (b_weights, d_weights) = if matrix.is_some() {
            (b_weights, d_weights)
        } else {
            let domain_len = self.plan.repacking().domain_len();
            (
                NativeRootWeights::new(domain_len, Vec::new(), b_weights.claim())?,
                NativeRootWeights::new(domain_len, Vec::new(), d_weights.claim())?,
            )
        };
        let span = tracing::info_span!("binary_root_weights_ordinary").entered();
        let ordinary_weights = match (&self.ordinary, ordinary_claim) {
            (Some(ordinary), Some((point, evaluation, payload))) => Some(ordinary.linear_weights(
                self.plan.repacking(),
                &self.setup,
                point,
                evaluation,
                payload,
                alpha,
                batches.get(ordinary_rows).ok_or(AkitaError::InvalidProof)?,
            )?),
            (None, None) => None,
            _ => return Err(AkitaError::InvalidProof),
        };
        drop(span);
        let _span = tracing::info_span!("binary_root_weights_merge").entered();
        let components = [
            Some(algebra),
            prime_weights,
            Some(b_weights),
            Some(d_weights),
            ordinary_weights,
        ];
        let linear = NativeRootWeights::merge(
            self.plan.repacking().domain_len(),
            components.into_iter().flatten(),
            projection_claim,
        )?;
        let mut negative_weights = Vec::new();
        negative_weights
            .try_reserve_exact(self.negative_binary_addresses.len())
            .map_err(|_| AkitaError::InvalidProof)?;
        for (&address, &batch) in self
            .negative_binary_addresses
            .iter()
            .zip(batches.get(negative_bits).ok_or(AkitaError::InvalidProof)?)
        {
            negative_weights.push((address, batch));
        }
        Ok((linear, negative_weights))
    }
}

#[cfg(test)]
#[path = "reference_weights.rs"]
pub(crate) mod reference;
