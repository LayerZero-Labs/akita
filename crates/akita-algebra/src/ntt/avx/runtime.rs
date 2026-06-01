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
    pub(super) avx512f: bool,
    pub(super) avx512dq: bool,
    pub(super) avx512bw: bool,
}

impl AvxCpuFeatures {
    #[inline]
    const fn has_avx512_ntt(self) -> bool {
        self.avx512f && self.avx512dq && self.avx512bw
    }
}

/// Return the enabled x86 CRT NTT SIMD mode, if any.
///
/// The result is cached because this function sits on hot dispatch boundaries.
pub fn avx_ntt_mode() -> Option<AvxNttMode> {
    static MODE: OnceLock<Option<AvxNttMode>> = OnceLock::new();
    *MODE.get_or_init(|| {
        select_avx_ntt_mode(
            std::env::var("AKITA_SCALAR_NTT").ok().as_deref(),
            std::env::var("AKITA_AVX_NTT").ok().as_deref(),
            std::env::var("AKITA_AVX512_NTT").ok().as_deref(),
            detect_cpu_features(),
        )
    })
}

/// Whether the host may use x86 `i32` transform kernels at all.
///
/// Both x86 modes gate on this; the chosen mode then selects the kernel shape.
/// `D = 32` always uses the dedicated small-degree AVX2 kernel. Wider degrees
/// use the width-aware `wide512` transform in `Avx512` mode, and the 128-bit
/// transform loop in this module in `Avx2` mode.
pub fn use_avx2_transform_ntt() -> bool {
    avx_ntt_mode().is_some() && std::is_x86_feature_detected!("avx2")
}

#[inline]
pub(super) fn select_avx_ntt_mode(
    scalar_ntt: Option<&str>,
    avx_ntt: Option<&str>,
    avx512_ntt: Option<&str>,
    cpu: AvxCpuFeatures,
) -> Option<AvxNttMode> {
    if scalar_ntt == Some("1") || avx_ntt == Some("0") {
        return None;
    }
    // AVX-512 is the default when available. `AKITA_AVX512_NTT=0` opts back out
    // to AVX2 for A/B comparison or hosts that downclock under AVX-512.
    if avx512_ntt != Some("0") && cpu.has_avx512_ntt() {
        return Some(AvxNttMode::Avx512);
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
        avx512f: std::is_x86_feature_detected!("avx512f"),
        avx512dq: std::is_x86_feature_detected!("avx512dq"),
        avx512bw: std::is_x86_feature_detected!("avx512bw"),
    }
}
