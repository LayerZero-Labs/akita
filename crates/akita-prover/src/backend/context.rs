use super::ProofContext;
use std::marker::PhantomData;
pub struct OperationCtx<'a, F, B: super::ProofScopeConsumer> {
    backend: &'a B,
    session: &'a B::ProofSessionHandle,
    context: ProofContext,
    field: PhantomData<F>,
}
impl<'a, F, B: super::ProofScopeConsumer> OperationCtx<'a, F, B> {
    pub fn new(backend: &'a B, session: &'a B::ProofSessionHandle, context: ProofContext) -> Self {
        Self {
            backend,
            session,
            context,
            field: PhantomData,
        }
    }
    pub fn backend(&self) -> &'a B {
        self.backend
    }
    pub fn proof_session(&self) -> &'a B::ProofSessionHandle {
        self.session
    }
    pub fn proof_context(&self) -> &ProofContext {
        &self.context
    }
    pub fn for_group(&self, group: usize) -> Self {
        Self::new(self.backend, self.session, self.context.for_group(group))
    }
}
