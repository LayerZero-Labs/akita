//! Shared public compilation of the binary root's fused linear relation.

use akita_challenges::Challenges;
use akita_error::{checked, AkitaError};
use akita_params::binary_setup::{derive_centered_q_matrix, BinaryFieldProfile};
use akita_params::{witness_commitment_domain_len, GroupCommitPhaseParams};
#[cfg(feature = "parallel")]
use jolt_field::solinas::parallel::*;
use jolt_field::{cfg_chunks, cfg_iter_mut, cfg_join, Field};

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
        entries.resize(columns, [0i128; 648]);
        cfg_iter_mut!(entries)
            .zip(cfg_chunks!(row, 648))
            .for_each(|(lifted, entry)| {
                for (output, &coefficient) in lifted.iter_mut().zip(entry) {
                    *output = i128::from(coefficient);
                }
            });
        matrix.push(entries);
    }
    if matrix.len() != manifest.a_rank() {
        return Err(AkitaError::InvalidProof);
    }
    Ok(matrix)
}

/// Admitted public inputs and non-A weights for one binary group.
///
/// The selected field profile determines coefficient bounds and batching.
/// Its outgoing coefficient map is authenticated by the ordinary successor commitment;
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
        prime.validate(manifest.descriptor())?;
        let domain_len = witness_commitment_domain_len(logical_len, destination_ring_dimension)?;
        let mut plan = BinaryRootWitnessPlan::for_profile(
            manifest.descriptor(),
            manifest.a_rank(),
            destination_ring_dimension,
            manifest.field_profile(),
            claim.is_some(),
            prime.planes().len(),
        )?;
        let ordinary = ordinary_profile
            .map(|profile| {
                OrdinaryNativeRootPlan::new(
                    profile,
                    plan.quotient_segment()
                        .unwrap_or(plan.repacking().segments().len()),
                )
            })
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
        let batch_count = checked::sum([
            plan.coefficient_batch_count()?,
            if claim.is_some() { 2 } else { 0 },
            checked::product([2, prime.planes().len()]).ok_or(AkitaError::InvalidProof)?,
            manifest.b_rows(),
            manifest
                .d_label(claim.is_some(), prime.planes().len())?
                .rank as usize,
            ordinary
                .as_ref()
                .map_or(0, OrdinaryNativeRootPlan::linear_batch_count),
            negative_binary_addresses.len(),
        ])
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

    /// Seven P128 projections or the P64 coefficient coordinates, then word
    /// parity/value, B/D, ordinary rows, negative bits and two batches per prime
    /// plane. No row
    /// reads a padding address, and none constrains one.
    pub fn batch_count(&self) -> usize {
        self.batch_count
    }

    /// Batch ranges: terminal, B, D, ordinary rows, negative bits, prime rows.
    fn batch_ranges(&self) -> Result<[Range<usize>; 6], AkitaError> {
        let terminal = checked::range(
            self.plan.coefficient_batch_count()?,
            if self.claim.is_some() { 2 } else { 0 },
        )
        .ok_or(AkitaError::InvalidProof)?;
        let b =
            checked::range(terminal.end, self.manifest.b_rows()).ok_or(AkitaError::InvalidProof)?;
        let d = checked::range(
            b.end,
            self.manifest
                .d_label(self.claim.is_some(), self.prime.planes().len())?
                .rank as usize,
        )
        .ok_or(AkitaError::InvalidProof)?;
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

    /// Interpret the profile's independent coordinates once. P64 has no J or y
    /// inputs: rho comes directly from the first a common batch coordinates.
    #[allow(clippy::type_complexity)]
    fn coefficient_inputs<'a, E: Field>(
        &self,
        batches: &[E],
        rho: &'a [[E; 648]],
        projected_values: &[i128],
    ) -> Result<(std::borrow::Cow<'a, [[E; 648]]>, E), AkitaError> {
        if self.manifest.field_profile() == BinaryFieldProfile::P128 {
            if projected_values.len() != super::BINARY_PROJECTION_COUNT {
                return Err(AkitaError::InvalidProof);
            }
            let mu = batches
                .get(..super::BINARY_PROJECTION_COUNT)
                .ok_or(AkitaError::InvalidProof)?;
            let claim = mu
                .iter()
                .zip(projected_values)
                .fold(E::zero(), |sum, (&mu, &y)| sum + mu * E::from_i128(y));
            return Ok((std::borrow::Cow::Borrowed(rho), claim));
        }
        if !rho.is_empty() || !projected_values.is_empty() {
            return Err(AkitaError::InvalidInput(
                "p64 relation has no J or y inputs".into(),
            ));
        }
        let coefficients = batches
            .get(..self.plan.coefficient_batch_count()?)
            .ok_or(AkitaError::InvalidProof)?;
        let mut rows = Vec::new();
        rows.try_reserve_exact(self.plan.projection_bounds().residual_rows())
            .map_err(|_| AkitaError::InvalidProof)?;
        for row in coefficients.chunks_exact(648) {
            rows.push(row.try_into().map_err(|_| AkitaError::InvalidProof)?);
        }
        Ok((std::borrow::Cow::Owned(rows), E::zero()))
    }

    /// Compile the single linear oracle after the outgoing commitment.
    ///
    /// Batch coordinates are independent field challenges, one per row.
    /// `commitment` is the public B payload; `d_payload`
    /// is the order-three commitment to the selected word and prime views.
    /// Proving supplies A; verification omits A and the B/D weights and
    /// authenticates their contribution through the typed prefix. For P128,
    /// `rho` batches the public J rows. For P64, supply empty `rho` and y slices;
    /// the independent coefficient weights come directly from `batches`.
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    pub fn relation<E: Field>(
        &self,
        challenges: &Challenges,
        alpha: E,
        batches: &[E],
        rho: &[[E; 648]],
        projected_values: &[i128],
        commitment: &[E],
        d_payload: &[E],
        ordinary_claim: Option<(&[E], E, &[E])>,
        matrix: Option<super::BinaryAOperand<'_, E>>,
    ) -> Result<(NativeRootWeights<E>, Vec<(usize, E)>), AkitaError> {
        if batches.len() != self.batch_count {
            return Err(AkitaError::InvalidSize {
                expected: self.batch_count,
                actual: batches.len(),
            });
        }
        let (rho, projection_claim) = self.coefficient_inputs(batches, rho, projected_values)?;
        let [terminal, b, d, ordinary_rows, negative_bits, prime_rows] = self.batch_ranges()?;
        // The A image and terminal, prime-plane and ordinary components are
        // independent and compiled in parallel. Their errors are checked in
        // the order in which they were compiled serially.
        let algebra = || {
            let _span = tracing::info_span!("binary_root_weights_algebra").entered();
            self.plan.relation_weights(
                self.claim.as_ref(),
                matrix,
                challenges,
                alpha,
                rho.as_ref(),
                batches.get(terminal).ok_or(AkitaError::InvalidProof)?,
            )
        };
        let prime_weights = || {
            let _span = tracing::info_span!("binary_root_weights_prime").entered();
            self.plan
                .prime_segment()
                .map(|segment| {
                    super::prime_plane_relation_weights(
                        self.manifest.descriptor(),
                        &self.prime,
                        self.plan.repacking(),
                        0,
                        segment,
                        challenges,
                        alpha,
                        batches.get(prime_rows).ok_or(AkitaError::InvalidProof)?,
                    )
                })
                .transpose()
        };
        let ordinary_weights = || {
            let _span = tracing::info_span!("binary_root_weights_ordinary").entered();
            match (&self.ordinary, ordinary_claim) {
                (Some(ordinary), Some((point, evaluation, payload))) => ordinary
                    .linear_weights(
                        self.plan.repacking(),
                        &self.setup,
                        point,
                        core::slice::from_ref(&evaluation),
                        payload,
                        alpha,
                        batches.get(ordinary_rows).ok_or(AkitaError::InvalidProof)?,
                    )
                    .map(Some),
                (None, None) => Ok(None),
                _ => Err(AkitaError::InvalidProof),
            }
        };
        let (algebra, (prime_weights, ordinary_weights)) =
            cfg_join!(algebra, || cfg_join!(prime_weights, ordinary_weights));
        let algebra = algebra?;
        let prime_weights = prime_weights?;
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
            self.manifest
                .d_label(self.claim.is_some(), self.prime.planes().len())?,
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
        let ordinary_weights = ordinary_weights?;
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
