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
        // The current AVX-512 transform plan intentionally retains AVX2
        // stages, and its target-feature contract requires AVX2 explicitly.
        self.avx2 && self.avx512f && self.avx512dq && self.avx512bw
    }
}

/// Return the enabled x86 CRT NTT SIMD mode, if any.
///
/// Pointwise kernels use the widest available mode by default. Transform
/// kernels have a separate policy because AVX2 is faster on the measured Ice
/// Lake host; set `AKITA_AVX512_NTT=1` to opt them into AVX-512 explicitly.
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
/// use true 256-bit AVX2 transforms by default, or the width-aware `wide512`
/// transform when AVX-512 is explicitly enabled.
pub fn use_avx2_transform_ntt() -> bool {
    avx_ntt_mode().is_some() && std::is_x86_feature_detected!("avx2")
}

/// Whether `i32` transforms should use their AVX-512 implementation.
///
/// AVX-512 remains available for explicit cross-machine benchmarking, but
/// true 256-bit AVX2 transforms are the default after winning every measured
/// D=64/128/256/512 transform on Ice Lake.
pub(crate) fn use_avx512_transform_ntt() -> bool {
    static USE_AVX512: OnceLock<bool> = OnceLock::new();
    *USE_AVX512.get_or_init(|| {
        select_avx512_transform_ntt(
            std::env::var("AKITA_AVX512_NTT").ok().as_deref(),
            avx_ntt_mode(),
        )
    })
}

#[inline]
pub(super) fn select_avx512_transform_ntt(
    avx512_ntt: Option<&str>,
    mode: Option<AvxNttMode>,
) -> bool {
    avx512_ntt == Some("1") && matches!(mode, Some(AvxNttMode::Avx512))
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
    // Pointwise kernels use AVX-512 by default when the complete current plan
    // is available. That plan also uses AVX2 transform stages.
    // `AKITA_AVX512_NTT=0` opts all kernels back out to AVX2.
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
