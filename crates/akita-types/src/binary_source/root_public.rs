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

/// Materialize the q24 source matrix from its canonical typed setup identity.
///
/// Matrix order is A row, source ring row, then natural coefficient. Both the
/// source commitment and the verifier's direct A evaluation use this function.
pub fn binary_root_matrix(
    seed: &[u8; 32],
    manifest: &BinaryCommittedGroupManifest,
) -> Result<Vec<Vec<[i128; 648]>>, AkitaError> {
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

/// Admitted public inputs and direct weights for one binary terminal claim.
///
/// This experimental root uses the fixed proof prime 2^128-275. Its outgoing
/// coefficient map is authenticated by the ordinary successor commitment;
/// source image digits, U, all carries, and every virtual zero share that map.
/// No security admission of the experimental source matrices is asserted.
#[derive(Clone, Debug)]
pub struct BinaryRootPublic {
    manifest: BinaryCommittedGroupManifest,
    claim: BinaryTerminalClaim,
    setup: AkitaSetupDescriptor,
    plan: BinaryRootWitnessPlan,
    matrix: Vec<Vec<[i128; 648]>>,
    ordinary: Option<OrdinaryNativeRootPlan>,
    negative_binary_addresses: Vec<usize>,
    padding_count: usize,
    batch_count: usize,
}

impl BinaryRootPublic {
    pub fn new(
        manifest: &BinaryCommittedGroupManifest,
        claim: &BinaryTerminalClaim,
        setup: &AkitaSetupDescriptor,
        logical_len: usize,
        destination_ring_dimension: usize,
        ordinary_profile: Option<GroupCommitPhaseParams>,
    ) -> Result<Self, AkitaError> {
        manifest.validate()?;
        claim.validate(manifest.descriptor())?;
        if manifest.descriptor().polynomials().len() != 1 {
            return Err(AkitaError::InvalidSetup(
                "binary root requires exactly one terminal polynomial".into(),
            ));
        }
        let domain_len = witness_commitment_domain_len(logical_len, destination_ring_dimension)?;
        let mut plan = BinaryRootWitnessPlan::new(
            manifest.descriptor(),
            manifest.a_rank(),
            destination_ring_dimension,
        )?;
        let ordinary = ordinary_profile
            .map(|profile| OrdinaryNativeRootPlan::new(profile, 6))
            .transpose()?;
        if let Some(profile) = ordinary_profile {
            plan = plan.with_ordinary(&profile)?;
        }
        let plan = plan.with_domain_len(domain_len)?;
        let negative_binary_addresses = ordinary
            .as_ref()
            .map(|ordinary| ordinary.negative_binary_addresses(plan.repacking()))
            .transpose()?
            .unwrap_or_default();
        let mut padding_count = 0usize;
        for address in 0..domain_len {
            let padding = plan.repacking().is_padding_address(address)?;
            if address >= logical_len && !padding {
                return Err(AkitaError::InvalidSetup(
                    "binary root natural digits exceed the outgoing witness".into(),
                ));
            }
            if padding {
                padding_count = checked::sum([padding_count, 1]).ok_or(AkitaError::InvalidProof)?;
            }
        }
        let batch_count = checked::product([2, manifest.b_rows()])
            .and_then(|authentication| {
                checked::sum([
                    manifest.a_rank(),
                    2,
                    authentication,
                    padding_count,
                    ordinary
                        .as_ref()
                        .map_or(0, OrdinaryNativeRootPlan::linear_batch_count),
                    negative_binary_addresses.len(),
                ])
            })
            .ok_or(AkitaError::InvalidProof)?;
        let matrix = binary_root_matrix(&setup.setup_seed.seed, manifest)?;
        Ok(Self {
            manifest: manifest.clone(),
            claim: claim.clone(),
            setup: setup.clone(),
            plan,
            matrix,
            ordinary,
            negative_binary_addresses,
            padding_count,
            batch_count,
        })
    }

    pub fn plan(&self) -> &BinaryRootWitnessPlan {
        &self.plan
    }

    pub fn ordinary_plan(&self) -> Option<&OrdinaryNativeRootPlan> {
        self.ordinary.as_ref()
    }

    /// A, parity/value, B/D, padding, ordinary rows, then negative-bit batches.
    pub fn batch_count(&self) -> usize {
        self.batch_count
    }

    /// Compile the single direct linear oracle after the outgoing commitment.
    ///
    /// Batch coordinates are independent field challenges, including one for
    /// each padding coefficient. `commitment` is the public B payload; `d_payload`
    /// is the order-three U commitment. The future setup adjoint replaces direct
    /// setup evaluation without changing this authenticated coefficient map.
    #[allow(clippy::type_complexity)]
    pub fn relation<E: Field>(
        &self,
        challenges: &Challenges,
        alpha: E,
        batches: &[E],
        commitment: &[E],
        d_payload: &[E],
        ordinary_claim: Option<(&[E], E, &[E])>,
    ) -> Result<(NativeRootWeights<E>, Vec<(usize, E)>), AkitaError> {
        if batches.len() != self.batch_count {
            return Err(AkitaError::InvalidSize {
                expected: self.batch_count,
                actual: batches.len(),
            });
        }
        let rank = self.manifest.a_rank();
        let terminal = checked::range(rank, 2).ok_or(AkitaError::InvalidProof)?;
        let b =
            checked::range(terminal.end, self.manifest.b_rows()).ok_or(AkitaError::InvalidProof)?;
        let d = checked::range(b.end, self.manifest.b_rows()).ok_or(AkitaError::InvalidProof)?;
        let padding = checked::range(d.end, self.padding_count).ok_or(AkitaError::InvalidProof)?;
        let ordinary_rows = checked::range(
            padding.end,
            self.ordinary
                .as_ref()
                .map_or(0, OrdinaryNativeRootPlan::linear_batch_count),
        )
        .ok_or(AkitaError::InvalidProof)?;
        let negative_bits = checked::range(ordinary_rows.end, self.negative_binary_addresses.len())
            .ok_or(AkitaError::InvalidProof)?;
        let terminal_batches: [E; 2] = batches
            .get(terminal)
            .ok_or(AkitaError::InvalidProof)?
            .try_into()
            .map_err(|_| AkitaError::InvalidProof)?;
        let algebra = self.plan.relation_weights(
            &self.claim,
            &self.matrix,
            challenges,
            alpha,
            batches.get(..rank).ok_or(AkitaError::InvalidProof)?,
            terminal_batches,
            batches.get(padding).ok_or(AkitaError::InvalidProof)?,
        )?;
        let b_weights = binary_digit_authentication_weights(
            &self.plan,
            1,
            &self.setup.setup_seed.seed,
            self.manifest.b_label(),
            commitment,
            alpha,
            batches.get(b).ok_or(AkitaError::InvalidProof)?,
        )?;
        let d_weights = binary_digit_authentication_weights(
            &self.plan,
            3,
            &self.setup.setup_seed.seed,
            SetupStreamLabel {
                role: SetupStreamRole::WitnessCommitment,
                ..self.manifest.b_label()
            },
            d_payload,
            alpha,
            batches.get(d).ok_or(AkitaError::InvalidProof)?,
        )?;
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
        let components = [
            Some(algebra),
            Some(b_weights),
            Some(d_weights),
            ordinary_weights,
        ];
        let count = checked::sum(
            components
                .iter()
                .flatten()
                .map(|weights| weights.terms().len()),
        )
        .ok_or(AkitaError::InvalidProof)?;
        let mut terms = Vec::new();
        terms
            .try_reserve_exact(count)
            .map_err(|_| AkitaError::InvalidProof)?;
        let mut claim = E::zero();
        for weights in components.iter().flatten() {
            terms.extend_from_slice(weights.terms());
            claim += weights.claim();
        }
        let linear = NativeRootWeights::new(self.plan.repacking().domain_len(), terms, claim)?;
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
