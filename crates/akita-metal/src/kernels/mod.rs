//! Embedded Metal kernel sources.

/// Metal source for `Fp128` vector arithmetic kernels.
pub const FP128_VECTOR_METAL: &str = include_str!("fp128_vector.metal");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp128_kernel_source_is_embedded() {
        assert!(FP128_VECTOR_METAL.contains("fp128_vector_add"));
        assert!(FP128_VECTOR_METAL.contains("fp128_vector_sub"));
        assert!(FP128_VECTOR_METAL.contains("fp128_vector_mul"));
    }
}
