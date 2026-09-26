//! Public operation context. Backends validate it against their private sessions.
use super::ProofScopeId;

/// Public identity and schedule position; this request confers no ownership.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProofContext {
    backend_id: u64,
    setup_digest: [u8; 32],
    scope_id: ProofScopeId,
    fold_level: u32,
    group_index: Option<usize>,
}

impl ProofContext {
    pub const fn new(
        backend_id: u64,
        setup_digest: [u8; 32],
        scope_id: ProofScopeId,
        fold_level: u32,
    ) -> Self {
        Self {
            backend_id,
            setup_digest,
            scope_id,
            fold_level,
            group_index: None,
        }
    }
    pub const fn for_group(self, group_index: usize) -> Self {
        Self {
            group_index: Some(group_index),
            ..self
        }
    }
    pub const fn backend_id(&self) -> u64 {
        self.backend_id
    }
    pub const fn setup_digest(&self) -> [u8; 32] {
        self.setup_digest
    }
    pub const fn scope_id(&self) -> ProofScopeId {
        self.scope_id
    }
    pub const fn fold_level(&self) -> u32 {
        self.fold_level
    }
    pub const fn group_index(&self) -> Option<usize> {
        self.group_index
    }
}
