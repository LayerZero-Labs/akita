//! Private backend identity and independent proof lifetimes.
use akita_error::AkitaError;
use akita_prover::backend::{ProofContext, ProofScopeId};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

static NEXT_BACKEND_ID: AtomicU64 = AtomicU64::new(1);

/// Owning proof lifetime. Numeric scope identifiers grant no cleanup authority.
pub struct CpuProofSessionHandle {
    owner: Arc<BackendIdentity>,
    proof: ScopeLease,
}

impl CpuProofSessionHandle {
    pub(crate) fn new(owner: Arc<BackendIdentity>, proof: ScopeLease) -> Self {
        Self { owner, proof }
    }

    pub(crate) fn scope_id(&self) -> ProofScopeId {
        self.proof.scope_id()
    }

    pub(crate) fn validate_owner(
        &self,
        owner: &Arc<BackendIdentity>,
    ) -> Result<&ScopeLease, AkitaError> {
        if self.owner.backend_id() != owner.backend_id() {
            return Err(AkitaError::InvalidInput(
                "proof session belongs to another backend".into(),
            ));
        }
        self.proof.validate_owner(owner)?;
        self.proof.validate(self.scope_id(), 0, None)?;
        Ok(&self.proof)
    }

    pub(crate) fn belongs_to(&self, owner: &Arc<BackendIdentity>) -> bool {
        self.owner.backend_id() == owner.backend_id()
    }

    pub(crate) fn scope_lease(&self) -> ScopeLease {
        self.proof.clone()
    }
}

impl Drop for CpuProofSessionHandle {
    fn drop(&mut self) {
        self.owner.abort_scope(&self.proof);
    }
}

impl core::fmt::Debug for CpuProofSessionHandle {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CpuProofSessionHandle")
            .finish_non_exhaustive()
    }
}

/// Only the owning proof scope can invalidate this operation lease.
#[derive(Clone)]
pub(crate) struct ScopeLease {
    state: Arc<ProofState>,
}

struct ProofState {
    backend_id: u64,
    setup_digest: [u8; 32],
    scope: ProofScopeId,
    active: AtomicBool,
    group_counts: Arc<[usize]>,
    commitments: Mutex<HashMap<(u32, usize), u128>>,
    plan: Option<(
        Arc<akita_types::FoldSchedule>,
        akita_types::OpeningClaimsLayout,
    )>,
}

impl ScopeLease {
    pub(crate) fn unbound() -> Self {
        Self {
            state: Arc::new(ProofState {
                backend_id: 0,
                setup_digest: [0; 32],
                scope: ProofScopeId::from_raw(0),
                active: AtomicBool::new(true),
                group_counts: vec![usize::MAX].into(),
                commitments: Mutex::new(HashMap::new()),
                plan: None,
            }),
        }
    }

    pub(crate) fn scope_id(&self) -> ProofScopeId {
        self.state.scope
    }

    pub(crate) fn is_active(&self) -> bool {
        self.state.active.load(Ordering::Acquire)
    }

    pub(crate) fn validate_owner(&self, owner: &BackendIdentity) -> Result<(), AkitaError> {
        if self.state.backend_id != owner.backend_id()
            || self.state.setup_digest != owner.setup_digest()
        {
            return Err(AkitaError::InvalidInput(
                "proof session belongs to another backend or setup".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn validate(
        &self,
        scope: ProofScopeId,
        fold_level: u32,
        group_index: Option<usize>,
    ) -> Result<(), AkitaError> {
        if scope != self.scope_id() || !self.is_active() {
            return Err(AkitaError::InvalidInput("inactive proof scope".into()));
        }
        let group_count = self
            .state
            .group_counts
            .get(fold_level as usize)
            .ok_or_else(|| {
                AkitaError::InvalidInput("fold level is outside the admitted proof".into())
            })?;
        if group_index.is_some_and(|group| group >= *group_count) {
            return Err(AkitaError::InvalidInput(
                "group is outside the admitted fold".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn validate_context(&self, context: &ProofContext) -> Result<(), AkitaError> {
        if context.backend_id() != self.state.backend_id
            || context.setup_digest() != self.state.setup_digest
        {
            return Err(AkitaError::InvalidInput(
                "operation belongs to another backend or setup".into(),
            ));
        }
        self.validate(
            context.scope_id(),
            context.fold_level(),
            context.group_index(),
        )
    }

    pub(crate) fn admit_commitment(
        &self,
        context: &ProofContext,
        commitment_id: u128,
    ) -> Result<(), AkitaError> {
        self.validate_context(context)?;
        let group = context.group_index().ok_or(AkitaError::InvalidProof)?;
        let mut commitments = self
            .state
            .commitments
            .lock()
            .map_err(|_| AkitaError::InvalidProof)?;
        let key = (context.fold_level(), group);
        if commitments.get(&key).is_some_and(|&id| id != commitment_id) {
            return Err(AkitaError::InvalidInput(
                "proof group already admitted a different commitment".into(),
            ));
        }
        commitments.insert(key, commitment_id);
        Ok(())
    }

    pub(crate) fn validate_commitment(
        &self,
        context: &ProofContext,
        commitment_id: u128,
    ) -> Result<(), AkitaError> {
        self.validate_context(context)?;
        let group = context.group_index().ok_or(AkitaError::InvalidProof)?;
        let commitments = self
            .state
            .commitments
            .lock()
            .map_err(|_| AkitaError::InvalidProof)?;
        #[cfg(test)]
        if self.state.plan.is_none() {
            return Ok(());
        }
        if commitments.get(&(context.fold_level(), group)) != Some(&commitment_id) {
            return Err(AkitaError::InvalidInput(
                "commitment was not admitted for this proof group".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn proof_plan(
        &self,
    ) -> Result<
        (
            Arc<akita_types::FoldSchedule>,
            akita_types::OpeningClaimsLayout,
        ),
        AkitaError,
    > {
        self.validate(self.scope_id(), 0, None)?;
        self.state
            .plan
            .clone()
            .ok_or_else(|| AkitaError::InvalidInput("proof has no admitted plan".into()))
    }

    fn finish(&self) -> Result<(), AkitaError> {
        self.state
            .active
            .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| ())
            .map_err(|_| {
                AkitaError::InvalidInput("proof scope is unknown or already finished".into())
            })
    }

    fn abort(&self) {
        self.state.active.store(false, Ordering::Release);
    }
}

pub(crate) struct BackendIdentity {
    backend_id: u64,
    setup_digest: [u8; 32],
    next_scope: AtomicU64,
    next_operation: AtomicU64,
}

impl BackendIdentity {
    pub(crate) fn new(setup_digest: [u8; 32]) -> Result<Arc<Self>, AkitaError> {
        let backend_id = NEXT_BACKEND_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| AkitaError::InvalidInput("backend identity exhausted".into()))?;
        Ok(Arc::new(Self {
            backend_id,
            setup_digest,
            next_scope: AtomicU64::new(1),
            next_operation: AtomicU64::new(1),
        }))
    }
    pub(crate) const fn backend_id(&self) -> u64 {
        self.backend_id
    }
    pub(crate) const fn setup_digest(&self) -> [u8; 32] {
        self.setup_digest
    }

    pub(crate) fn begin_proof(
        &self,
        plan: &akita_types::FoldSchedule,
        layout: &akita_types::OpeningClaimsLayout,
    ) -> Result<ScopeLease, AkitaError> {
        plan.validate_structure()?;
        let mut group_counts = Vec::with_capacity(plan.recursive_folds.len() + 2);
        group_counts.push(layout.num_groups());
        for fold in &plan.recursive_folds {
            group_counts.push(fold.params.groups().len());
        }
        group_counts.push(1);
        self.insert_scope(group_counts, Some((Arc::new(plan.clone()), layout.clone())))
    }
    #[cfg(test)]
    fn begin_scope(&self) -> Result<ScopeLease, AkitaError> {
        self.insert_scope(vec![1], None)
    }
    #[cfg(test)]
    pub(crate) fn begin_test_scope(
        &self,
        group_counts: Vec<usize>,
    ) -> Result<ScopeLease, AkitaError> {
        self.insert_scope(group_counts, None)
    }

    fn insert_scope(
        &self,
        group_counts: Vec<usize>,
        plan: Option<(
            Arc<akita_types::FoldSchedule>,
            akita_types::OpeningClaimsLayout,
        )>,
    ) -> Result<ScopeLease, AkitaError> {
        let sequence = self
            .next_scope
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| AkitaError::InvalidInput("proof scope identity exhausted".into()))?;
        let scope =
            ProofScopeId::from_raw((u128::from(self.backend_id) << 64) | u128::from(sequence));
        Ok(ScopeLease {
            state: Arc::new(ProofState {
                backend_id: self.backend_id,
                setup_digest: self.setup_digest,
                scope,
                active: AtomicBool::new(true),
                group_counts: group_counts.into(),
                commitments: Mutex::new(HashMap::new()),
                plan,
            }),
        })
    }
    pub(crate) fn finish_scope(&self, proof: &ScopeLease) -> Result<(), AkitaError> {
        proof.validate_owner(self)?;
        proof.finish()
    }
    pub(crate) fn abort_scope(&self, proof: &ScopeLease) {
        if proof.validate_owner(self).is_ok() {
            proof.abort();
        }
    }
    pub(crate) fn next_operation_id(&self) -> Result<u128, AkitaError> {
        self.next_operation
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map(u128::from)
            .map_err(|_| AkitaError::InvalidInput("operation identity exhausted".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn proof_session_drop_and_finish_require_the_owning_capability() {
        use akita_prover::backend::ProofScopeConsumer;
        let backend = crate::CpuBackend::<
            jolt_field::Prime128OffsetA7F7,
            jolt_field::Prime128OffsetA7F7,
        >::for_arithmetic_tests();
        let foreign = crate::CpuBackend::<
            jolt_field::Prime128OffsetA7F7,
            jolt_field::Prime128OffsetA7F7,
        >::for_arithmetic_tests();
        let a = backend.owner().begin_scope().unwrap();
        let b = backend.owner().begin_scope().unwrap();
        let lease_a = a.clone();
        let lease_b = b.clone();
        let first = CpuProofSessionHandle::new(Arc::clone(backend.owner()), a);
        let second = CpuProofSessionHandle::new(Arc::clone(backend.owner()), b);
        assert!(foreign.finish_scope(&first).is_err());
        foreign.abort_scope_best_effort(&first);
        assert!(lease_a.is_active());
        drop(first);
        assert!(!lease_a.is_active());
        assert!(lease_b.is_active());
        backend.finish_scope(&second).unwrap();
        assert!(!lease_b.is_active());
        assert!(backend.finish_scope(&second).is_err());
        drop(second);
    }

    #[test]
    fn ordered_commitment_admission_is_scope_local_and_immutable() {
        let owner = BackendIdentity::new([8; 32]).unwrap();
        let a = owner.begin_test_scope(vec![2]).unwrap();
        let b = owner.begin_test_scope(vec![2]).unwrap();
        let context = ProofContext::new(owner.backend_id(), owner.setup_digest(), a.scope_id(), 0)
            .for_group(0);
        a.admit_commitment(&context, 7).unwrap();
        a.admit_commitment(&context, 7).unwrap();
        assert!(a.admit_commitment(&context, 8).is_err());
        a.admit_commitment(&context.for_group(1), 8).unwrap();
        let independent =
            ProofContext::new(owner.backend_id(), owner.setup_digest(), b.scope_id(), 0)
                .for_group(0);
        b.admit_commitment(&independent, 8).unwrap();
        owner.finish_scope(&a).unwrap();
        assert!(a.admit_commitment(&context, 7).is_err());
        b.admit_commitment(&independent, 8).unwrap();
    }
    #[test]
    fn independent_scopes_survive_completion_and_abort() {
        let owner = BackendIdentity::new([7; 32]).unwrap();
        let a = owner.begin_scope().unwrap();
        let b = owner.begin_scope().unwrap();
        owner.finish_scope(&a).unwrap();
        assert!(!a.is_active());
        assert!(b.is_active());
        assert!(owner.finish_scope(&a).is_err());
        owner.abort_scope(&b);
        assert!(!b.is_active());
    }
    #[test]
    fn foreign_owner_and_setup_cannot_authorize_context() {
        let a = BackendIdentity::new([3; 32]).unwrap();
        let b = BackendIdentity::new([3; 32]).unwrap();
        let proof = a.begin_scope().unwrap();
        let context = ProofContext::new(a.backend_id(), a.setup_digest(), proof.scope_id(), 0);
        assert!(proof.validate_owner(&a).is_ok());
        assert!(proof.validate_owner(&b).is_err());
        assert!(proof.validate_context(&context).is_ok());
        assert!(proof
            .validate_context(&ProofContext::new(
                a.backend_id(),
                [9; 32],
                proof.scope_id(),
                0,
            ))
            .is_err());
        a.abort_scope(&proof);
        assert!(proof.validate_context(&context).is_err());
    }
    #[test]
    fn independent_proofs_can_finish_on_different_threads() {
        let owner = BackendIdentity::new([4; 32]).unwrap();
        let proofs = (0..4)
            .map(|_| owner.begin_scope().unwrap())
            .collect::<Vec<_>>();
        let leases = proofs.clone();
        std::thread::scope(|threads| {
            for proof in proofs {
                let owner = &owner;
                threads.spawn(move || owner.finish_scope(&proof).unwrap());
            }
        });
        assert!(leases.iter().all(|lease| !lease.is_active()));
    }
}
