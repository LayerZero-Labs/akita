//! Declarative Jolt entrypoints for the Akita recursion benchmark.
//!
//! [`integration`] owns artifact decoding, the trusted benchmark boundary,
//! prepared cache installation, statement and transcript construction,
//! verifier execution, cycle phases, and status mapping. Each declaration here
//! remains a distinct Jolt program identity because its config and field are
//! monomorphized into a separate guest function.

mod integration;

use akita_config::proof_optimized::{fp128, fp32, fp64};
use akita_config::RecursiveCommitmentConfig;
use akita_recursion_glue::AkitaJoltCase;
use integration::declare_akita_guest;

declare_akita_guest!(
    akita_verify_fp32,
    AkitaJoltCase::OneHotFp32,
    fp32::OneHot,
    2048
);
declare_akita_guest!(
    akita_verify_fp64,
    AkitaJoltCase::OneHotFp64,
    fp64::OneHot,
    512
);
declare_akita_guest!(
    akita_verify_fp128_direct,
    AkitaJoltCase::OneHotFp128Direct,
    fp128::OneHot,
    512
);
declare_akita_guest!(
    akita_verify_fp128_recursive,
    AkitaJoltCase::OneHotFp128Recursive,
    RecursiveCommitmentConfig<fp128::OneHot>,
    512
);
declare_akita_guest!(
    akita_verify,
    AkitaJoltCase::OneHotFp128MultiGroupRecursive,
    RecursiveCommitmentConfig<fp128::OneHot>,
    512
);
