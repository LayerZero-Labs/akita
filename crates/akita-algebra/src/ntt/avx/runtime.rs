use std::sync::OnceLock;

/// Runtime-selected x86 CRT NTT SIMD mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AvxNttMode {
    /// AVX2 kernels using 256-bit integer vectors.
    Avx2,
    /// AVX-512 kernels using 512-bit integer vectors.
    Avx512,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct AvxCpuFeatures {
    pub(super) avx2: bool,
}

/// Return the enabled x86 CRT NTT SIMD mode, if any.
///
/// AVX2 is the production backend after winning the measured end-to-end i32
/// workloads on Ice Lake. `AKITA_SCALAR_NTT=1` disables it globally.
/// The result is cached because this function sits on hot dispatch boundaries.
pub fn avx_ntt_mode() -> Option<AvxNttMode> {
    static MODE: OnceLock<Option<AvxNttMode>> = OnceLock::new();
    *MODE.get_or_init(|| {
        select_avx_ntt_mode(
            std::env::var("AKITA_SCALAR_NTT").ok().as_deref(),
            detect_cpu_features(),
        )
    })
}

/// Whether the host may use x86 `i32` transform kernels at all.
///
/// `D = 32` uses the dedicated small-degree kernel; wider degrees use the
/// measured 256-bit AVX2 transform.
pub fn use_avx2_transform_ntt() -> bool {
    avx_ntt_mode().is_some() && std::is_x86_feature_detected!("avx2")
}

#[inline]
pub(super) fn select_avx_ntt_mode(
    scalar_ntt: Option<&str>,
    cpu: AvxCpuFeatures,
) -> Option<AvxNttMode> {
    if scalar_ntt == Some("1") {
        return None;
    }
    if cpu.avx2 {
        return Some(AvxNttMode::Avx2);
    }
    None
}

#[inline]
pub(super) fn detect_cpu_features() -> AvxCpuFeatures {
    AvxCpuFeatures {
        avx2: std::is_x86_feature_detected!("avx2"),
    }
}
