//! Backend selection occurs once per plan, never inside an arithmetic loop.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Backend {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512VbmiBatch,
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    Neon,
}

pub(super) fn selected_backend() -> Backend {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::ntt::avx::use_avx2_transform_ntt() {
            if vbmi_available() {
                Backend::Avx512VbmiBatch
            } else {
                Backend::Avx2
            }
        } else {
            Backend::Scalar
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        if crate::ntt::neon::use_neon_ntt() {
            Backend::Neon
        } else {
            Backend::Scalar
        }
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    Backend::Scalar
}

#[cfg(target_arch = "x86_64")]
pub(super) fn vbmi_available() -> bool {
    std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512vl")
        && std::is_x86_feature_detected!("avx512bw")
        && std::is_x86_feature_detected!("avx512vbmi")
        && std::is_x86_feature_detected!("avx512bitalg")
}

#[cfg(test)]
pub(super) fn available_backends() -> Vec<Backend> {
    #[cfg_attr(
        not(any(
            target_arch = "x86_64",
            all(target_arch = "aarch64", target_feature = "neon")
        )),
        expect(unused_mut)
    )]
    let mut backends = vec![Backend::Scalar];
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        backends.push(Backend::Avx2);
    }
    #[cfg(target_arch = "x86_64")]
    if vbmi_available() {
        backends.push(Backend::Avx512VbmiBatch);
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    backends.push(Backend::Neon);
    backends
}
