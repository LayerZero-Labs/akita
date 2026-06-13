//! Runtime-to-const-generic dispatch shared by prover and verifier.

/// Bridge a runtime ring dimension to a const-generic `D` context.
///
/// Returns an [`AkitaError`](akita_field::AkitaError) instead of panicking so it
/// is safe to use across verifier-reachable paths.
#[macro_export]
macro_rules! dispatch_ring_dim_result {
    ($d:expr, |$D:ident| $body:expr) => {{
        let __d = $d;
        match __d {
            32 => {
                const $D: usize = 32;
                $body
            }
            64 => {
                const $D: usize = 64;
                $body
            }
            128 => {
                const $D: usize = 128;
                $body
            }
            256 => {
                const $D: usize = 256;
                $body
            }
            _ => Err(akita_field::AkitaError::InvalidInput(format!(
                "unsupported ring dimension: {__d}"
            ))),
        }
    }};
}

#[cfg(test)]
mod tests {
    use akita_field::AkitaError;

    #[test]
    fn dispatch_ring_dim_result_accepts_supported_dimensions() {
        for d in [32usize, 64, 128, 256] {
            let got: Result<usize, AkitaError> = crate::dispatch_ring_dim_result!(d, |D| Ok(D));
            assert_eq!(got.expect("supported ring dimension"), d);
        }
    }

    #[test]
    fn dispatch_ring_dim_result_rejects_unsupported_dimensions() {
        let err: AkitaError = crate::dispatch_ring_dim_result!(16usize, |D| Ok(D))
            .expect_err("unsupported ring dimension must be rejected");
        assert!(matches!(err, AkitaError::InvalidInput(_)));
    }
}
