//! x86 runtime dispatch helpers for CRT NTT SIMD kernels.
//!
//! `AKITA_SCALAR_NTT=1` forces the scalar fallback for all CRT NTT SIMD.
//! `AKITA_AVX_NTT=0` disables only x86 CRT NTT SIMD. `AKITA_AVX512_NTT=1`
//! opts into AVX-512 kernels when the host supports the required features.

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
struct AvxCpuFeatures {
    avx2: bool,
    avx512f: bool,
    avx512dq: bool,
    avx512bw: bool,
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

#[inline]
fn select_avx_ntt_mode(
    scalar_ntt: Option<&str>,
    avx_ntt: Option<&str>,
    avx512_ntt: Option<&str>,
    cpu: AvxCpuFeatures,
) -> Option<AvxNttMode> {
    if scalar_ntt == Some("1") || avx_ntt == Some("0") {
        return None;
    }
    if avx512_ntt == Some("1") && cpu.has_avx512_ntt() {
        return Some(AvxNttMode::Avx512);
    }
    if cpu.avx2 {
        return Some(AvxNttMode::Avx2);
    }
    None
}

#[inline]
fn detect_cpu_features() -> AvxCpuFeatures {
    AvxCpuFeatures {
        avx2: std::is_x86_feature_detected!("avx2"),
        avx512f: std::is_x86_feature_detected!("avx512f"),
        avx512dq: std::is_x86_feature_detected!("avx512dq"),
        avx512bw: std::is_x86_feature_detected!("avx512bw"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const AVX2_ONLY: AvxCpuFeatures = AvxCpuFeatures {
        avx2: true,
        avx512f: false,
        avx512dq: false,
        avx512bw: false,
    };

    const AVX512_CAPABLE: AvxCpuFeatures = AvxCpuFeatures {
        avx2: true,
        avx512f: true,
        avx512dq: true,
        avx512bw: true,
    };

    #[test]
    fn avx_mode_defaults_to_avx2_when_supported() {
        assert_eq!(
            select_avx_ntt_mode(None, None, None, AVX2_ONLY),
            Some(AvxNttMode::Avx2)
        );
    }

    #[test]
    fn avx512_is_opt_in() {
        assert_eq!(
            select_avx_ntt_mode(None, None, None, AVX512_CAPABLE),
            Some(AvxNttMode::Avx2)
        );
        assert_eq!(
            select_avx_ntt_mode(None, None, Some("1"), AVX512_CAPABLE),
            Some(AvxNttMode::Avx512)
        );
    }

    #[test]
    fn scalar_kill_switch_precedes_avx_flags() {
        assert_eq!(
            select_avx_ntt_mode(Some("1"), None, Some("1"), AVX512_CAPABLE),
            None
        );
    }

    #[test]
    fn avx_kill_switch_disables_x86_ntt_simd() {
        assert_eq!(
            select_avx_ntt_mode(None, Some("0"), Some("1"), AVX512_CAPABLE),
            None
        );
    }

    #[test]
    fn avx512_opt_in_falls_back_to_avx2_without_full_features() {
        let missing_bw = AvxCpuFeatures {
            avx512bw: false,
            ..AVX512_CAPABLE
        };
        assert_eq!(
            select_avx_ntt_mode(None, None, Some("1"), missing_bw),
            Some(AvxNttMode::Avx2)
        );
    }
}
