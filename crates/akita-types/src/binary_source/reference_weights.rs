//! Test-only reference: the flat term-list construction of the root weights.
//!
//! These are the compilers that preceded the structured families, kept term
//! for term, without padding rows. Each pushes one `(address, weight)` pair
//! per public coefficient weight. The differential tests compare the
//! structured evaluation and the prover's dense table with them.

use akita_algebra::ring::trinomial::adjoint::TrinomialAdjoint;
use akita_algebra::ring::trinomial::module_map::pack_signed_components;
use akita_algebra::ring::trinomial::MinusTrinomial;
use akita_algebra::ring::{shifted_remainder_evaluations, PlusTrinomial, ResidueKernelPoint};
use akita_challenges::Challenges;
use akita_error::{checked, AkitaError};
use akita_params::binary_setup::{SetupStreamLabel, SetupStreamRole, BINARY_SETUP_P};
use akita_params::RelationPolynomial;
use jolt_field::Field;

use super::super::authentication::{setup_entry, validate_label};
use super::super::{
    binary_scalar_challenges, binary_scalar_lift, BinaryEvaluationBounds, BinaryEvaluationSegments,
    BinaryRootPublic, BinaryRootWitnessPlan, BinarySourceDescriptor, BinaryTerminalClaim,
    OrdinaryNativeRootPlan, PrimePlaneManifest, BINARY_DIGIT_COMMITMENT_DEGREE,
    BINARY_PARITY_CARRY_DIGITS, BINARY_VALUE_CARRY_DIGITS, CANONICAL_IMAGE_WINDOW,
    PRIME_PLANE_DIGITS,
};
use crate::native_root_relation::NativeRootWeights;
use crate::root_repacking::RootCoefficientRepacking;

pub(crate) fn direct_a_weights<E: Field>(
    plan: &BinaryRootWitnessPlan,
    matrix: Option<&[Vec<[i128; 648]>]>,
    challenges: &Challenges,
    rho: &[[E; 648]],
) -> Result<NativeRootWeights<E>, AkitaError> {
    let descriptor = plan.descriptor();
    let columns = descriptor.index_map().columns();
    let rows = checked::product([4, columns])
        .and_then(|d| checked::exact_div(descriptor.index_map().padded_carriers(), d))
        .ok_or(AkitaError::InvalidProof)?;
    let owners = descriptor.polynomials().len();
    if rho.len() != checked::product([owners, plan.rank()]).ok_or(AkitaError::InvalidProof)? {
        return Err(AkitaError::InvalidInput(
            "A coefficient functional has the wrong rank".into(),
        ));
    }
    if let Some(matrix) = matrix {
        if matrix.len() != plan.rank()
            || matrix.iter().any(|row| row.len() != rows)
            || matrix.iter().flatten().flatten().any(|&c| {
                c < i128::from(CANONICAL_IMAGE_WINDOW.0) || c > i128::from(CANONICAL_IMAGE_WINDOW.1)
            })
        {
            return Err(AkitaError::InvalidInput(
                "A differs from its centered typed geometry".into(),
            ));
        }
    }
    let embedded = binary_scalar_challenges(descriptor, challenges)?
        .into_iter()
        .map(|scalar| {
            Ok(pack_signed_components(&[scalar, [0; 162], [0; 162], [0; 162]])?.map(E::from_i128))
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    let adjoint = TrinomialAdjoint::<648, MinusTrinomial>::new()?;
    let response_digits = plan
        .repacking()
        .segments()
        .first()
        .ok_or(AkitaError::InvalidProof)?
        .digits;
    let image_digits = plan
        .repacking()
        .segments()
        .get(1)
        .ok_or(AkitaError::InvalidProof)?
        .digits;
    let response_terms = if matrix.is_some() {
        checked::product([owners, plan.rank(), rows, 648, response_digits])
            .ok_or(AkitaError::InvalidProof)?
    } else {
        0
    };
    let image_terms = checked::product([owners, plan.rank(), columns, 648, image_digits])
        .ok_or(AkitaError::InvalidProof)?;
    let term_count = checked::sum([response_terms, image_terms]).ok_or(AkitaError::InvalidProof)?;
    let mut terms = Vec::new();
    terms
        .try_reserve_exact(term_count)
        .map_err(|_| AkitaError::InvalidProof)?;
    for owner in 0..owners {
        for i in 0..plan.rank() {
            let row_index =
                checked::mul_add(owner, plan.rank(), i).ok_or(AkitaError::InvalidProof)?;
            let weights = rho.get(row_index).ok_or(AkitaError::InvalidProof)?;
            if let Some(matrix) = matrix {
                for (j, a) in matrix
                    .get(i)
                    .ok_or(AkitaError::InvalidProof)?
                    .iter()
                    .enumerate()
                {
                    let element =
                        checked::mul_add(owner, rows, j).ok_or(AkitaError::InvalidProof)?;
                    for (t, w) in adjoint
                        .apply(weights, &a.map(E::from_i128))?
                        .into_iter()
                        .enumerate()
                    {
                        terms.extend(plan.repacking().coefficient_weights(0, element, t, w)?);
                    }
                }
            }
            for column in 0..columns {
                let challenge_index =
                    checked::mul_add(owner, columns, column).ok_or(AkitaError::InvalidProof)?;
                let challenge = embedded
                    .get(challenge_index)
                    .ok_or(AkitaError::InvalidProof)?;
                let element = checked::mul_add(challenge_index, plan.rank(), i)
                    .ok_or(AkitaError::InvalidProof)?;
                for (t, w) in adjoint.apply(weights, challenge)?.into_iter().enumerate() {
                    terms.extend(plan.repacking().coefficient_weights(1, element, t, -w)?);
                }
            }
        }
    }
    NativeRootWeights::new(plan.repacking().domain_len(), terms, E::zero())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn terminal_weights<E: Field>(
    claim: &BinaryTerminalClaim,
    descriptor: &BinarySourceDescriptor,
    repacking: &RootCoefficientRepacking,
    segments: BinaryEvaluationSegments,
    challenges: &Challenges,
    alpha: E,
    batches: [E; 2],
    proof_modulus: u128,
) -> Result<NativeRootWeights<E>, AkitaError> {
    let (row_weights, column_weights) = claim.equality_weights(descriptor)?;
    let columns = descriptor.index_map().columns();
    let scalar_rows = row_weights.len();
    BinaryEvaluationBounds::new(scalar_rows, columns, proof_modulus)?;
    let ring_rows = checked::exact_div(scalar_rows, 4).ok_or(AkitaError::InvalidProof)?;
    let owner = descriptor
        .polynomials()
        .iter()
        .position(|id| id == claim.polynomial())
        .ok_or(AkitaError::InvalidProof)?;
    let polynomials = descriptor.polynomials().len();
    let expected_response =
        checked::product([polynomials, ring_rows]).ok_or(AkitaError::InvalidProof)?;
    let expected_partials =
        checked::product([polynomials, columns]).ok_or(AkitaError::InvalidProof)?;
    let response_shape = repacking
        .segments()
        .get(segments.response)
        .ok_or(AkitaError::InvalidProof)?;
    if response_shape.elements != expected_response
        || response_shape.digits != 4
        || response_shape.geometry.polynomial() != RelationPolynomial::minus_trinomial(648)?
        || response_shape.geometry.coordinate_plane_count() != 1
    {
        return Err(AkitaError::InvalidProof);
    }
    for (segment, elements, digits) in [
        (segments.partials, expected_partials, 1),
        (
            segments.parity_carry,
            polynomials,
            BINARY_PARITY_CARRY_DIGITS,
        ),
        (segments.value_carry, polynomials, BINARY_VALUE_CARRY_DIGITS),
    ] {
        let shape = repacking
            .segments()
            .get(segment)
            .ok_or(AkitaError::InvalidProof)?;
        if shape.elements != elements
            || shape.digits != digits
            || shape.geometry.polynomial() != RelationPolynomial::plus_trinomial(162)?
            || shape.geometry.coordinate_plane_count() != 1
        {
            return Err(AkitaError::InvalidProof);
        }
    }
    let scalar_challenges = binary_scalar_challenges(descriptor, challenges)?;
    let mut terms = Vec::new();
    for (row, weight) in row_weights.into_iter().enumerate() {
        let multiplier = binary_scalar_lift(weight).map(E::from_i128);
        let mut kernel = [E::zero(); 162];
        shifted_remainder_evaluations::<E, PlusTrinomial>(&multiplier, alpha, &mut kernel)
            .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
        let element =
            checked::mul_add(owner, ring_rows, row / 4).ok_or(AkitaError::InvalidProof)?;
        for (coefficient, weight) in kernel.into_iter().enumerate() {
            let packed =
                checked::mul_add(4, coefficient, row % 4).ok_or(AkitaError::InvalidProof)?;
            let address = repacking.coefficient_address(segments.response, element, 0, packed)?;
            let sign = if coefficient % 2 == 0 {
                E::one()
            } else {
                -E::one()
            };
            terms.push((address, batches[0] * sign * weight));
        }
    }
    for (column, weight) in column_weights.into_iter().enumerate() {
        let challenge_index =
            checked::mul_add(owner, columns, column).ok_or(AkitaError::InvalidProof)?;
        let challenge = scalar_challenges
            .get(challenge_index)
            .ok_or(AkitaError::InvalidProof)?;
        let multiplier = binary_scalar_lift(weight).map(E::from_i128);
        let mut value_kernel = [E::zero(); 162];
        let mut parity_kernel = [E::zero(); 162];
        shifted_remainder_evaluations::<E, PlusTrinomial>(&multiplier, alpha, &mut value_kernel)
            .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
        shifted_remainder_evaluations::<E, PlusTrinomial>(
            &challenge.map(E::from_i128),
            alpha,
            &mut parity_kernel,
        )
        .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
        for (coefficient, (value, parity)) in
            value_kernel.into_iter().zip(parity_kernel).enumerate()
        {
            terms.extend(repacking.coefficient_weights(
                segments.partials,
                challenge_index,
                coefficient,
                batches[1] * value - batches[0] * parity,
            )?);
        }
    }
    let mut power = E::one();
    for coefficient in 0..162 {
        for (segment, batch) in [
            (segments.parity_carry, batches[0]),
            (segments.value_carry, batches[1]),
        ] {
            terms.extend(repacking.coefficient_weights(
                segment,
                owner,
                coefficient,
                -E::from_u64(2) * batch * power,
            )?);
        }
        power *= alpha;
    }
    let value = akita_algebra::binary::BinaryField162::from_bytes(claim.value())
        .ok_or(AkitaError::InvalidProof)?;
    let target = binary_scalar_lift(value)
        .into_iter()
        .rev()
        .fold(E::zero(), |acc, coefficient| {
            acc * alpha + E::from_i128(coefficient)
        });
    NativeRootWeights::new(repacking.domain_len(), terms, batches[1] * target)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prime_plane_weights<E: Field>(
    descriptor: &BinarySourceDescriptor,
    manifest: &PrimePlaneManifest,
    repacking: &RootCoefficientRepacking,
    response_segment: usize,
    plane_segment: usize,
    challenges: &Challenges,
    alpha: E,
    batches: &[E],
) -> Result<NativeRootWeights<E>, AkitaError> {
    if E::from_u128(BINARY_SETUP_P) != E::zero()
        || batches.len()
            != checked::product([manifest.planes().len(), 2]).ok_or(AkitaError::InvalidProof)?
    {
        return Err(AkitaError::InvalidProof);
    }
    let columns = descriptor.index_map().columns();
    let scalar_rows = checked::exact_div(descriptor.index_map().padded_carriers(), columns)
        .ok_or(AkitaError::InvalidProof)?;
    let ring_rows = checked::exact_div(scalar_rows, 4).ok_or(AkitaError::InvalidProof)?;
    for (segment, elements, digits, polynomial) in [
        (
            response_segment,
            checked::product([descriptor.polynomials().len(), ring_rows]),
            4,
            RelationPolynomial::minus_trinomial(648)?,
        ),
        (
            plane_segment,
            checked::product([manifest.planes().len(), columns]),
            PRIME_PLANE_DIGITS,
            RelationPolynomial::plus_trinomial(162)?,
        ),
    ] {
        let shape = repacking
            .segments()
            .get(segment)
            .ok_or(AkitaError::InvalidProof)?;
        if Some(shape.elements) != elements
            || shape.digits != digits
            || shape.geometry.polynomial() != polynomial
            || shape.geometry.coordinate_plane_count() != 1
        {
            return Err(AkitaError::InvalidProof);
        }
    }
    let scalar_challenges = binary_scalar_challenges(descriptor, challenges)?;
    let term_count = checked::sum([
        checked::product([scalar_rows, 162, 4]).ok_or(AkitaError::InvalidProof)?,
        checked::product([columns, 162, PRIME_PLANE_DIGITS]).ok_or(AkitaError::InvalidProof)?,
    ])
    .and_then(|count| checked::product([manifest.planes().len(), count]))
    .ok_or(AkitaError::InvalidProof)?;
    let mut terms = Vec::new();
    terms
        .try_reserve_exact(term_count)
        .map_err(|_| AkitaError::InvalidProof)?;
    let mut target = E::zero();
    for (plane_index, claim) in manifest.planes().iter().enumerate() {
        let (rows, column_weights, omega) = claim.weights::<E>(descriptor)?;
        let batch_index = checked::product([plane_index, 2]).ok_or(AkitaError::InvalidProof)?;
        let relation_batch = *batches.get(batch_index).ok_or(AkitaError::InvalidProof)?;
        let reading_batch = *batches
            .get(checked::sum([batch_index, 1]).ok_or(AkitaError::InvalidProof)?)
            .ok_or(AkitaError::InvalidProof)?;
        for (row, row_weight) in rows.into_iter().enumerate() {
            if row_weight == E::zero() {
                continue;
            }
            let element = checked::mul_add(claim.polynomial, ring_rows, row / 4)
                .ok_or(AkitaError::InvalidProof)?;
            let mut power = relation_batch * row_weight;
            for coefficient in 0..162 {
                let packed =
                    checked::mul_add(coefficient, 4, row % 4).ok_or(AkitaError::InvalidProof)?;
                let signed = if coefficient % 2 == 0 { power } else { -power };
                terms.extend(repacking.coefficient_weights(
                    response_segment,
                    element,
                    packed,
                    signed,
                )?);
                power *= alpha;
            }
        }
        for (column, column_weight) in column_weights.into_iter().enumerate() {
            let challenge_index = checked::mul_add(claim.polynomial, columns, column)
                .ok_or(AkitaError::InvalidProof)?;
            let multiplier = scalar_challenges
                .get(challenge_index)
                .ok_or(AkitaError::InvalidProof)?
                .map(E::from_i128);
            let mut kernel = [E::zero(); 162];
            shifted_remainder_evaluations::<E, PlusTrinomial>(&multiplier, alpha, &mut kernel)
                .map_err(|error| AkitaError::Internal(error.to_string()))?;
            let element =
                checked::mul_add(plane_index, columns, column).ok_or(AkitaError::InvalidProof)?;
            for (coefficient, relation) in kernel.into_iter().enumerate() {
                let weight = reading_batch
                    * column_weight
                    * *omega.get(coefficient).ok_or(AkitaError::InvalidProof)?
                    - relation_batch * relation;
                terms.extend(repacking.coefficient_weights(
                    plane_segment,
                    element,
                    coefficient,
                    weight,
                )?);
            }
        }
        target += reading_batch * E::from_u128(claim.value);
    }
    NativeRootWeights::new(repacking.domain_len(), terms, target)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn digit_authentication_weights<E: Field>(
    plan: &BinaryRootWitnessPlan,
    segments: &[usize],
    seed: &[u8; 32],
    label: SetupStreamLabel,
    payload: &[E],
    alpha: E,
    row_batches: &[E],
) -> Result<NativeRootWeights<E>, AkitaError> {
    let rows = validate_label::<E>(label)?;
    let expected_segments = match label.role {
        SetupStreamRole::ImageCommitment => vec![1],
        SetupStreamRole::WitnessCommitment => plan.opening_segments(),
        _ => return Err(AkitaError::InvalidProof),
    };
    if segments != expected_segments {
        return Err(AkitaError::InvalidSetup(
            "binary digit authentication role differs from its segments".into(),
        ));
    }
    let expected =
        checked::product([rows, BINARY_DIGIT_COMMITMENT_DEGREE]).ok_or(AkitaError::InvalidProof)?;
    if payload.len() != expected || row_batches.len() != rows {
        return Err(AkitaError::InvalidProof);
    }
    let mut addresses = Vec::new();
    for &segment in segments {
        let shape = plan
            .repacking()
            .segments()
            .get(segment)
            .ok_or(AkitaError::InvalidProof)?;
        let width = checked::product([
            shape.geometry.polynomial_modulus_dimension(),
            shape.geometry.coordinate_plane_count(),
        ])
        .ok_or(AkitaError::InvalidProof)?;
        let count = checked::product([shape.elements, shape.digits, width])
            .ok_or(AkitaError::InvalidProof)?;
        addresses
            .try_reserve_exact(count)
            .map_err(|_| AkitaError::InvalidProof)?;
        for element in 0..shape.elements {
            for digit in 0..shape.digits {
                for coefficient in 0..width {
                    addresses.push(plan.repacking().coefficient_address(
                        segment,
                        element,
                        digit,
                        coefficient,
                    )?);
                }
            }
        }
    }
    let count = addresses.len();
    let columns =
        checked::div_ceil(count, BINARY_DIGIT_COMMITMENT_DEGREE).ok_or(AkitaError::InvalidProof)?;
    let term_count = checked::product([rows, count]).ok_or(AkitaError::InvalidProof)?;
    let mut terms = Vec::new();
    terms
        .try_reserve_exact(term_count)
        .map_err(|_| AkitaError::InvalidProof)?;
    let mut claim = E::zero();
    let kernel = ResidueKernelPoint::new(alpha, BINARY_DIGIT_COMMITMENT_DEGREE)?;
    for (row, (target, &batch)) in payload
        .chunks_exact(BINARY_DIGIT_COMMITMENT_DEGREE)
        .zip(row_batches)
        .enumerate()
    {
        claim += batch
            * target
                .iter()
                .rev()
                .fold(E::zero(), |sum, &coefficient| sum * alpha + coefficient);
        let row = u32::try_from(row).map_err(|_| AkitaError::InvalidProof)?;
        for column in 0..columns {
            let matrix = setup_entry(seed, label, row, column)?;
            let mut weights = [E::zero(); BINARY_DIGIT_COMMITMENT_DEGREE];
            kernel.field_kernel_into(matrix.coefficients(), &mut weights)?;
            for (coefficient, &weight) in weights.iter().enumerate() {
                let compact = checked::mul_add(column, BINARY_DIGIT_COMMITMENT_DEGREE, coefficient)
                    .ok_or(AkitaError::InvalidProof)?;
                if compact >= count {
                    break;
                }
                let address = *addresses.get(compact).ok_or(AkitaError::InvalidProof)?;
                terms.push((address, batch * weight));
            }
        }
    }
    NativeRootWeights::new(plan.repacking().domain_len(), terms, claim)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn plan_weights<E: Field>(
    plan: &BinaryRootWitnessPlan,
    claim: Option<&BinaryTerminalClaim>,
    matrix: Option<&[Vec<[i128; 648]>]>,
    challenges: &Challenges,
    alpha: E,
    rho: &[[E; 648]],
    terminal_batches: impl AsRef<[E]>,
) -> Result<NativeRootWeights<E>, AkitaError> {
    let a = direct_a_weights(plan, matrix, challenges, rho)?;
    let Some(claim) = claim else {
        if !terminal_batches.as_ref().is_empty() {
            return Err(AkitaError::InvalidProof);
        }
        return Ok(a);
    };
    let terminal = terminal_weights(
        claim,
        plan.descriptor(),
        plan.repacking(),
        BinaryEvaluationSegments {
            response: 0,
            partials: 2,
            parity_carry: 3,
            value_carry: 4,
        },
        challenges,
        alpha,
        terminal_batches
            .as_ref()
            .try_into()
            .map_err(|_| AkitaError::InvalidProof)?,
        BINARY_SETUP_P,
    )?;
    NativeRootWeights::merge(plan.repacking().domain_len(), [a, terminal], E::zero())
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub(crate) fn root_relation<E: Field>(
    public: &BinaryRootPublic,
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
    if batches.len() != public.batch_count {
        return Err(AkitaError::InvalidSize {
            expected: public.batch_count,
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
    let terminal = checked::range(7, if public.claim.is_some() { 2 } else { 0 })
        .ok_or(AkitaError::InvalidProof)?;
    let b =
        checked::range(terminal.end, public.manifest.b_rows()).ok_or(AkitaError::InvalidProof)?;
    let d = checked::range(b.end, public.manifest.b_rows()).ok_or(AkitaError::InvalidProof)?;
    let ordinary_rows = checked::range(
        d.end,
        public
            .ordinary
            .as_ref()
            .map_or(0, OrdinaryNativeRootPlan::linear_batch_count),
    )
    .ok_or(AkitaError::InvalidProof)?;
    let negative_bits = checked::range(ordinary_rows.end, public.negative_binary_addresses.len())
        .ok_or(AkitaError::InvalidProof)?;
    let prime_rows = checked::range(
        negative_bits.end,
        checked::product([2, public.prime.planes().len()]).ok_or(AkitaError::InvalidProof)?,
    )
    .ok_or(AkitaError::InvalidProof)?;
    let algebra = plan_weights(
        &public.plan,
        public.claim.as_ref(),
        matrix,
        challenges,
        alpha,
        rho,
        batches.get(terminal).ok_or(AkitaError::InvalidProof)?,
    )?;
    let prime_weights = if let Some(segment) = public.plan.prime_segment() {
        Some(prime_plane_weights(
            public.manifest.descriptor(),
            &public.prime,
            public.plan.repacking(),
            0,
            segment,
            challenges,
            alpha,
            batches.get(prime_rows).ok_or(AkitaError::InvalidProof)?,
        )?)
    } else {
        None
    };
    let b_weights = digit_authentication_weights(
        &public.plan,
        &[1],
        &public.setup.setup_seed.seed,
        public.manifest.b_label(),
        commitment,
        alpha,
        batches.get(b).ok_or(AkitaError::InvalidProof)?,
    )?;
    let d_weights = digit_authentication_weights(
        &public.plan,
        &public.plan.opening_segments(),
        &public.setup.setup_seed.seed,
        SetupStreamLabel {
            role: SetupStreamRole::WitnessCommitment,
            ..public.manifest.b_label()
        },
        d_payload,
        alpha,
        batches.get(d).ok_or(AkitaError::InvalidProof)?,
    )?;
    // Without A the relation is the verifier's: the typed prefix carries B/D.
    let (b_weights, d_weights) = if matrix.is_some() {
        (b_weights, d_weights)
    } else {
        let domain_len = public.plan.repacking().domain_len();
        (
            NativeRootWeights::new(domain_len, Vec::new(), b_weights.claim())?,
            NativeRootWeights::new(domain_len, Vec::new(), d_weights.claim())?,
        )
    };
    let ordinary_weights = match (&public.ordinary, ordinary_claim) {
        (Some(ordinary), Some((point, evaluation, payload))) => Some(ordinary.linear_weights(
            public.plan.repacking(),
            &public.setup,
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
        prime_weights,
        Some(b_weights),
        Some(d_weights),
        ordinary_weights,
    ];
    let linear = NativeRootWeights::merge(
        public.plan.repacking().domain_len(),
        components.into_iter().flatten(),
        projection_claim,
    )?;
    let mut negative_weights = Vec::new();
    negative_weights
        .try_reserve_exact(public.negative_binary_addresses.len())
        .map_err(|_| AkitaError::InvalidProof)?;
    for (&address, &batch) in public
        .negative_binary_addresses
        .iter()
        .zip(batches.get(negative_bits).ok_or(AkitaError::InvalidProof)?)
    {
        negative_weights.push((address, batch));
    }
    Ok((linear, negative_weights))
}

#[cfg(test)]
#[path = "reference_tests.rs"]
mod tests;
