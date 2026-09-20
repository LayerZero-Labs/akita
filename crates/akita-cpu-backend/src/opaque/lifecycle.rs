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
    scope: ProofScopeId,
}

impl CpuProofSessionHandle {
    pub(crate) fn new(owner: Arc<BackendIdentity>, scope: ProofScopeId) -> Self {
        Self { owner, scope }
    }

    pub(crate) const fn scope_id(&self) -> ProofScopeId {
        self.scope
    }

    pub(crate) fn validate_owner(
        &self,
        owner: &Arc<BackendIdentity>,
    ) -> Result<ProofScopeId, AkitaError> {
        if self.owner.backend_id() != owner.backend_id() {
            return Err(AkitaError::InvalidInput(
                "proof session belongs to another backend".into(),
            ));
        }
        owner.validate_scope(self.scope)?;
        Ok(self.scope)
    }

    pub(crate) fn belongs_to(&self, owner: &Arc<BackendIdentity>) -> bool {
        self.owner.backend_id() == owner.backend_id()
    }
}

impl Drop for CpuProofSessionHandle {
    fn drop(&mut self) {
        self.owner.abort_scope(self.scope);
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
pub(crate) struct ScopeLease(Arc<AtomicBool>);

impl ScopeLease {
    pub(crate) fn is_active(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

pub(crate) struct BackendIdentity {
    backend_id: u64,
    setup_digest: [u8; 32],
    next_scope: AtomicU64,
    next_operation: AtomicU64,
    active: Mutex<HashMap<ProofScopeId, ActiveProof>>,
}

struct ActiveProof {
    lease: Arc<AtomicBool>,
    group_counts: Vec<usize>,
    commitments: HashMap<(u32, usize), u128>,
    plan: Option<(
        Arc<akita_types::FoldSchedule>,
        akita_types::OpeningClaimsLayout,
    )>,
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
            active: Mutex::new(HashMap::new()),
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
    ) -> Result<ProofScopeId, AkitaError> {
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
    fn begin_scope(&self) -> Result<ProofScopeId, AkitaError> {
        self.insert_scope(vec![1], None)
    }
    #[cfg(test)]
    pub(crate) fn begin_test_scope(
        &self,
        group_counts: Vec<usize>,
    ) -> Result<ProofScopeId, AkitaError> {
        self.insert_scope(group_counts, None)
    }

    fn insert_scope(
        &self,
        group_counts: Vec<usize>,
        plan: Option<(
            Arc<akita_types::FoldSchedule>,
            akita_types::OpeningClaimsLayout,
        )>,
    ) -> Result<ProofScopeId, AkitaError> {
        let sequence = self
            .next_scope
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| AkitaError::InvalidInput("proof scope identity exhausted".into()))?;
        let scope =
            ProofScopeId::from_raw((u128::from(self.backend_id) << 64) | u128::from(sequence));
        self.active
            .lock()
            .map_err(|_| AkitaError::InvalidInput("proof scope registry unavailable".into()))?
            .insert(
                scope,
                ActiveProof {
                    lease: Arc::new(AtomicBool::new(true)),
                    group_counts,
                    commitments: HashMap::new(),
                    plan,
                },
            );
        Ok(scope)
    }
    pub(crate) fn scope_lease(&self, scope: ProofScopeId) -> Result<ScopeLease, AkitaError> {
        if scope.raw() >> 64 != u128::from(self.backend_id) {
            return Err(AkitaError::InvalidInput(
                "proof scope belongs to another backend".into(),
            ));
        }
        self.active
            .lock()
            .map_err(|_| AkitaError::InvalidInput("proof scope registry unavailable".into()))?
            .get(&scope)
            .map(|proof| ScopeLease(proof.lease.clone()))
            .ok_or_else(|| AkitaError::InvalidInput("inactive proof scope".into()))
    }
    pub(crate) fn validate_scope(&self, scope: ProofScopeId) -> Result<(), AkitaError> {
        if !self.scope_lease(scope)?.is_active() {
            return Err(AkitaError::InvalidInput("inactive proof scope".into()));
        }
        Ok(())
    }
    pub(crate) fn admit_commitment(
        &self,
        context: &ProofContext,
        commitment_id: u128,
    ) -> Result<(), AkitaError> {
        self.validate_context(context)?;
        let group = context.group_index().ok_or(AkitaError::InvalidProof)?;
        let mut active = self.active.lock().map_err(|_| AkitaError::InvalidProof)?;
        let proof = active
            .get_mut(&context.scope_id())
            .ok_or(AkitaError::InvalidProof)?;
        let key = (context.fold_level(), group);
        if proof
            .commitments
            .get(&key)
            .is_some_and(|&id| id != commitment_id)
        {
            return Err(AkitaError::InvalidInput(
                "proof group already admitted a different commitment".into(),
            ));
        }
        proof.commitments.insert(key, commitment_id);
        Ok(())
    }
    pub(crate) fn validate_commitment(
        &self,
        context: &ProofContext,
        commitment_id: u128,
    ) -> Result<(), AkitaError> {
        self.validate_context(context)?;
        let group = context.group_index().ok_or(AkitaError::InvalidProof)?;
        let active = self.active.lock().map_err(|_| AkitaError::InvalidProof)?;
        let proof = active
            .get(&context.scope_id())
            .ok_or(AkitaError::InvalidProof)?;
        #[cfg(test)]
        if proof.plan.is_none() {
            return Ok(());
        }
        if proof.commitments.get(&(context.fold_level(), group)) != Some(&commitment_id) {
            return Err(AkitaError::InvalidInput(
                "commitment was not admitted for this proof group".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn proof_plan(
        &self,
        scope: ProofScopeId,
    ) -> Result<
        (
            Arc<akita_types::FoldSchedule>,
            akita_types::OpeningClaimsLayout,
        ),
        AkitaError,
    > {
        self.validate_scope(scope)?;
        self.active
            .lock()
            .map_err(|_| AkitaError::InvalidInput("proof scope registry unavailable".into()))?
            .get(&scope)
            .and_then(|proof| proof.plan.clone())
            .ok_or_else(|| AkitaError::InvalidInput("proof has no admitted plan".into()))
    }
    pub(crate) fn validate_context(&self, context: &ProofContext) -> Result<(), AkitaError> {
        if context.backend_id() != self.backend_id || context.setup_digest() != self.setup_digest {
            return Err(AkitaError::InvalidInput(
                "operation belongs to another backend or setup".into(),
            ));
        }
        self.validate_scope(context.scope_id())?;
        let active = self
            .active
            .lock()
            .map_err(|_| AkitaError::InvalidInput("proof scope registry unavailable".into()))?;
        let proof = active
            .get(&context.scope_id())
            .ok_or_else(|| AkitaError::InvalidInput("inactive proof scope".into()))?;
        let count = proof
            .group_counts
            .get(context.fold_level() as usize)
            .ok_or_else(|| {
                AkitaError::InvalidInput("fold level is outside the admitted proof".into())
            })?;
        if context.group_index().is_some_and(|group| group >= *count) {
            return Err(AkitaError::InvalidInput(
                "group is outside the admitted fold".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn finish_scope(&self, scope: ProofScopeId) -> Result<(), AkitaError> {
        if scope.raw() >> 64 != u128::from(self.backend_id) {
            return Err(AkitaError::InvalidInput(
                "proof scope belongs to another backend".into(),
            ));
        }
        let lease = self
            .active
            .lock()
            .map_err(|_| AkitaError::InvalidInput("proof scope registry unavailable".into()))?
            .remove(&scope)
            .ok_or_else(|| {
                AkitaError::InvalidInput("proof scope is unknown or already finished".into())
            })?;
        lease.lease.store(false, Ordering::Release);
        Ok(())
    }
    pub(crate) fn abort_scope(&self, scope: ProofScopeId) {
        let mut active = self
            .active
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if let Some(proof) = active.remove(&scope) {
            proof.lease.store(false, Ordering::Release);
        }
    }
    pub(crate) fn next_operation_id(&self) -> Result<u128, AkitaError> {
        self.next_operation
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map(u128::from)
            .map_err(|_| AkitaError::InvalidInput("operation identity exhausted".into()))
    }
}

impl Drop for BackendIdentity {
    fn drop(&mut self) {
        let active = self
            .active
            .get_mut()
            .unwrap_or_else(|poison| poison.into_inner());
        for proof in active.values() {
            proof.lease.store(false, Ordering::Release);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn proof_session_drop_and_finish_require_the_owning_capability() {
        use akita_prover::backend::ProofScopeConsumer;
        let backend = crate::CpuBackend::for_arithmetic_tests();
        let foreign = crate::CpuBackend::for_arithmetic_tests();
        let a = backend.owner().begin_scope().unwrap();
        let b = backend.owner().begin_scope().unwrap();
        let lease_a = backend.owner().scope_lease(a).unwrap();
        let lease_b = backend.owner().scope_lease(b).unwrap();
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
        let context =
            ProofContext::new(owner.backend_id(), owner.setup_digest(), a, 0).for_group(0);
        owner.admit_commitment(&context, 7).unwrap();
        owner.admit_commitment(&context, 7).unwrap();
        assert!(owner.admit_commitment(&context, 8).is_err());
        owner.admit_commitment(&context.for_group(1), 8).unwrap();
        let independent =
            ProofContext::new(owner.backend_id(), owner.setup_digest(), b, 0).for_group(0);
        owner.admit_commitment(&independent, 8).unwrap();
        owner.finish_scope(a).unwrap();
        assert!(owner.admit_commitment(&context, 7).is_err());
        owner.admit_commitment(&independent, 8).unwrap();
    }
    #[test]
    fn independent_scopes_survive_completion_and_abort() {
        let owner = BackendIdentity::new([7; 32]).unwrap();
        let a = owner.begin_scope().unwrap();
        let b = owner.begin_scope().unwrap();
        let lease_a = owner.scope_lease(a).unwrap();
        let lease_b = owner.scope_lease(b).unwrap();
        owner.finish_scope(a).unwrap();
        assert!(!lease_a.is_active());
        assert!(lease_b.is_active());
        assert!(owner.finish_scope(a).is_err());
        owner.abort_scope(b);
        assert!(!lease_b.is_active());
    }
    #[test]
    fn foreign_owner_and_setup_cannot_authorize_context() {
        let a = BackendIdentity::new([3; 32]).unwrap();
        let b = BackendIdentity::new([3; 32]).unwrap();
        let scope = a.begin_scope().unwrap();
        let context = ProofContext::new(a.backend_id(), a.setup_digest(), scope, 0);
        assert!(a.validate_context(&context).is_ok());
        assert!(b.validate_context(&context).is_err());
        assert!(a
            .validate_context(&ProofContext::new(a.backend_id(), [9; 32], scope, 0))
            .is_err());
        a.abort_scope(scope);
        assert!(a.validate_context(&context).is_err());
    }
    #[test]
    fn independent_proofs_can_finish_on_different_threads() {
        let owner = BackendIdentity::new([4; 32]).unwrap();
        let scopes = (0..4)
            .map(|_| owner.begin_scope().unwrap())
            .collect::<Vec<_>>();
        let leases = scopes
            .iter()
            .map(|&scope| owner.scope_lease(scope).unwrap())
            .collect::<Vec<_>>();
        std::thread::scope(|threads| {
            for scope in scopes {
                let owner = &owner;
                threads.spawn(move || owner.finish_scope(scope).unwrap());
            }
        });
        assert!(leases.iter().all(|lease| !lease.is_active()));
    }
}
