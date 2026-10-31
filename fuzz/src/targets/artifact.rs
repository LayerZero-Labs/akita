//! Trusted-schedule artifact admission from mutated artifact bytes.
//!
//! Seeds are the shipped `.aks` files. Admission must reject malformed,
//! tampered, or wrongly bound artifacts with an error, never a panic, and an
//! admitted artifact must be canonical: re-encoding reproduces its bytes.

use crate::input::Reader;
use crate::stats;
use akita_config::proof_optimized::{fp128, fp32, fp64};
use akita_config::{CommitmentConfig, RecursiveCommitmentConfig, TrustedScheduleCatalog};
use akita_params::{ChunkedWitnessCfg, MultiChunkProfileId};

/// Trusted caller policy for the shipped families, independent of fuzzed bytes.
pub fn chunking(family: &str) -> ChunkedWitnessCfg {
    match family {
        "fp128_dense_multi_chunk"
        | "fp128_onehot_multi_chunk"
        | "fp128_onehot_recursive_multi_chunk_w8r2" => MultiChunkProfileId::W8R2.cfg(),
        "fp128_onehot_multi_chunk_w2r2" => MultiChunkProfileId::W2R2.cfg(),
        "fp128_onehot_multi_chunk_w4r2" => MultiChunkProfileId::W4R2.cfg(),
        "fp128_onehot_multi_chunk_w2r3" | "fp128_onehot_recursive_multi_chunk_w2r3" => {
            MultiChunkProfileId::W2R3.cfg()
        }
        "fp128_onehot_multi_chunk_w4r3" | "fp128_onehot_recursive_multi_chunk_w4r3" => {
            MultiChunkProfileId::W4R3.cfg()
        }
        "fp128_dense_multi_chunk_w8r3"
        | "fp128_onehot_multi_chunk_w8r3"
        | "fp128_onehot_recursive_multi_chunk_w8r3" => MultiChunkProfileId::W8R3.cfg(),
        "fp128_dense"
        | "fp128_dense_bounded"
        | "fp128_onehot"
        | "fp128_dense_recursive"
        | "fp128_onehot_recursive"
        | "fp128_onehot_k16_recursive"
        | "fp32_dense"
        | "fp32_onehot"
        | "fp32_dense_recursive"
        | "fp64_dense"
        | "fp64_onehot"
        | "fp64_dense_recursive" => ChunkedWitnessCfg::default_non_chunked(),
        _ => panic!("unregistered fuzz catalog family: {family}"),
    }
}

fn admit<Cfg: CommitmentConfig>(bytes: &[u8]) {
    let Ok(catalog) = TrustedScheduleCatalog::<Cfg>::from_artifact_bytes(
        bytes,
        chunking(Cfg::schedule_family_name()),
    ) else {
        stats::count("artifact_rejected");
        return;
    };
    let encoded = catalog
        .catalog()
        .to_artifact_bytes()
        .unwrap_or_else(|error| panic!("admitted catalog fails to re-encode: {error:?}"));
    assert_eq!(encoded, bytes, "admitted artifact is not canonical");
    assert!(!catalog.is_empty(), "admitted catalogs are never empty");
    stats::count("artifact_admitted");
}

pub fn run(data: &[u8]) {
    let mut reader = Reader::new(data);
    let selector = reader.u8();
    let bytes = reader.rest();
    match selector as usize % FAMILIES.len() {
        0 => admit::<fp128::Dense>(bytes),
        1 => admit::<fp128::DenseBounded>(bytes),
        2 => admit::<fp128::DenseMultiChunk>(bytes),
        3 => admit::<fp128::OneHot>(bytes),
        4 => admit::<fp128::OneHotMultiChunk>(bytes),
        5 => admit::<fp128::OneHotMultiChunkW2R2>(bytes),
        6 => admit::<fp128::OneHotMultiChunkW4R2>(bytes),
        7 => admit::<RecursiveCommitmentConfig<fp128::Dense>>(bytes),
        8 => admit::<RecursiveCommitmentConfig<fp128::OneHot>>(bytes),
        9 => admit::<RecursiveCommitmentConfig<fp128::OneHotMultiChunk>>(bytes),
        10 => admit::<fp32::Dense>(bytes),
        11 => admit::<fp32::OneHot>(bytes),
        12 => admit::<RecursiveCommitmentConfig<fp32::Dense>>(bytes),
        13 => admit::<fp64::Dense>(bytes),
        14 => admit::<fp64::OneHot>(bytes),
        15 => admit::<RecursiveCommitmentConfig<fp64::Dense>>(bytes),
        16 => admit::<fp128::OneHotMultiChunkW2R3>(bytes),
        17 => admit::<fp128::OneHotMultiChunkW4R3>(bytes),
        18 => admit::<fp128::OneHotMultiChunkW8R3>(bytes),
        19 => admit::<fp128::DenseMultiChunkW8R3>(bytes),
        20 => admit::<RecursiveCommitmentConfig<fp128::OneHotMultiChunkW2R3>>(bytes),
        21 => admit::<RecursiveCommitmentConfig<fp128::OneHotMultiChunkW4R3>>(bytes),
        22 => admit::<RecursiveCommitmentConfig<fp128::OneHotMultiChunkW8R3>>(bytes),
        _ => admit::<RecursiveCommitmentConfig<fp128::OneHotK16>>(bytes),
    }
}

/// Selector byte for each shipped family, in the order used by [`run`].
pub const FAMILIES: [&str; 24] = [
    "fp128_dense",
    "fp128_dense_bounded",
    "fp128_dense_multi_chunk",
    "fp128_onehot",
    "fp128_onehot_multi_chunk",
    "fp128_onehot_multi_chunk_w2r2",
    "fp128_onehot_multi_chunk_w4r2",
    "fp128_dense_recursive",
    "fp128_onehot_recursive",
    "fp128_onehot_recursive_multi_chunk_w8r2",
    "fp32_dense",
    "fp32_onehot",
    "fp32_dense_recursive",
    "fp64_dense",
    "fp64_onehot",
    "fp64_dense_recursive",
    "fp128_onehot_multi_chunk_w2r3",
    "fp128_onehot_multi_chunk_w4r3",
    "fp128_onehot_multi_chunk_w8r3",
    "fp128_dense_multi_chunk_w8r3",
    "fp128_onehot_recursive_multi_chunk_w2r3",
    "fp128_onehot_recursive_multi_chunk_w4r3",
    "fp128_onehot_recursive_multi_chunk_w8r3",
    "fp128_onehot_k16_recursive",
];
