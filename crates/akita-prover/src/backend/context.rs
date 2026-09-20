use super::ProofContext;
use std::marker::PhantomData;
pub struct OperationCtx<'a, F, B> {
    backend: &'a B,
    context: ProofContext,
    field: PhantomData<F>,
}
impl<'a, F, B> OperationCtx<'a, F, B> {
    pub fn new(backend: &'a B, context: ProofContext) -> Self {
        Self {
            backend,
            context,
            field: PhantomData,
        }
    }
    pub fn backend(&self) -> &'a B {
        self.backend
    }
    pub fn proof_context(&self) -> &ProofContext {
        &self.context
    }
    pub fn for_group(&self, group: usize) -> Self {
        Self::new(self.backend, self.context.for_group(group))
    }
}
