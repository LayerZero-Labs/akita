use jolt_field::Field;
#[derive(Debug, Clone, Copy)]
pub(crate) struct ValidatedWitnessOpeningPlan<'a, E: Field> {
    point: &'a [E],
    ring_dimension: usize,
    witness_len: usize,
}

/// Validated native opening geometry for one opaque recursive witness.
impl<'a, E: Field> ValidatedWitnessOpeningPlan<'a, E> {
    #[allow(dead_code)]
    pub(crate) const fn new(point: &'a [E], ring_dimension: usize, witness_len: usize) -> Self {
        Self {
            point,
            ring_dimension,
            witness_len,
        }
    }
    pub(crate) const fn point(&self) -> &'a [E] {
        self.point
    }
    pub(crate) const fn ring_dimension(&self) -> usize {
        self.ring_dimension
    }
    pub(crate) const fn witness_len(&self) -> usize {
        self.witness_len
    }
}

/// Validated public inputs for one consumer-owned witness EOR session.
#[derive(Debug, Clone)]
pub(crate) struct ValidatedWitnessEorPlan<'a, E: Field> {
    claim_coefficients: &'a [E],
    tail_point: &'a [E],
    eta: &'a [E],
    extra_point: Vec<E>,
    input_claim: E,
    ring_dimension: usize,
}

impl<'a, E: Field> ValidatedWitnessEorPlan<'a, E> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        claim_coefficients: &'a [E],
        tail_point: &'a [E],
        eta: &'a [E],
        extra_point: Vec<E>,
        input_claim: E,
        ring_dimension: usize,
    ) -> Self {
        Self {
            claim_coefficients,
            tail_point,
            eta,
            extra_point,
            input_claim,
            ring_dimension,
        }
    }
    pub(crate) const fn claim_coefficients(&self) -> &'a [E] {
        self.claim_coefficients
    }
    pub(crate) const fn tail_point(&self) -> &'a [E] {
        self.tail_point
    }
    pub(crate) const fn eta(&self) -> &'a [E] {
        self.eta
    }
    pub(crate) fn extra_point(&self) -> &[E] {
        &self.extra_point
    }
    pub(crate) fn input_claim(&self) -> E {
        self.input_claim
    }
    pub(crate) const fn ring_dimension(&self) -> usize {
        self.ring_dimension
    }
}

/// Scalar openings and the private state retained for their EOR computation.
pub(crate) struct PreparedWitnessOpening<E: Field, H> {
    messages: Vec<E>,
    partials: Vec<E>,
    handle: H,
}
impl<E: Field, H> PreparedWitnessOpening<E, H> {
    pub(crate) fn new(messages: Vec<E>, partials: Vec<E>, handle: H) -> Self {
        Self {
            messages,
            partials,
            handle,
        }
    }
    pub(crate) fn into_parts(self) -> (Vec<E>, Vec<E>, H) {
        (self.messages, self.partials, self.handle)
    }
}
