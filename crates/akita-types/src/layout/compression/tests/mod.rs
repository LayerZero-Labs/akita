mod catalog;
mod terminal;
use super::*;
use akita_challenges::SparseChallengeConfig;
use akita_field::{Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59};

use crate::schedule::PrecommittedGroupParams;
use crate::{PolynomialGroupLayout, PrecommittedLevelParams, DEFAULT_SIS_SECURITY_BITS};

const D: usize = 32;

fn key(family: SisModulusFamily, d: usize, raw_bound: u128, col_len: usize) -> AjtaiKeyParams {
    let table_key =
        sis_table_key_for_linf_bound(DEFAULT_SIS_SECURITY_BITS, family, d as u32, raw_bound)
            .expect("test SIS row");
    AjtaiKeyParams::try_new_with_min_rank(table_key, col_len).expect("test secure key")
}

fn level() -> LevelParams {
    let mut lp = LevelParams::params_only(
        SisModulusFamily::Q128,
        64,
        6,
        1,
        1,
        1,
        SparseChallengeConfig::pm1_only(64),
    )
    .with_decomp(1, 1, 1, 1, 0)
    .unwrap();
    lp.b_key = key(SisModulusFamily::Q128, D, 63, 1);
    lp.d_key = key(SisModulusFamily::Q128, D, 63, 1);
    lp
}

fn chain_for(
    lp: &LevelParams,
    source: CompressionSourceId,
    source_key: &AjtaiKeyParams,
    alphabets: &[CompressionAlphabet],
) -> CompressionChainSpec {
    chain_for_profile(
        source,
        source_key,
        alphabets,
        SisModulusFamily::Q128,
        128,
        D,
        lp.log_basis,
    )
}

fn chain_for_profile(
    source: CompressionSourceId,
    source_key: &AjtaiKeyParams,
    alphabets: &[CompressionAlphabet],
    family: SisModulusFamily,
    field_bits: u32,
    map_d: usize,
    range_log_basis: u32,
) -> CompressionChainSpec {
    let source_d = source_key.sis_table_key().ring_dimension as usize;
    let mut previous_output = source_key.row_len() * source_d;
    let maps = alphabets
        .iter()
        .copied()
        .map(|alphabet| {
            let depth = alphabet_facts(alphabet, field_bits, range_log_basis).unwrap();
            let raw_bound = match alphabet {
                CompressionAlphabet::NegativeBinary => 1,
                CompressionAlphabet::OpeningBase { .. } => (1u128 << range_log_basis) - 1,
            };
            let input = previous_output * depth;
            assert_eq!(input % map_d, 0);
            let key = key(family, map_d, raw_bound, input / map_d);
            previous_output = key.row_len() * map_d;
            CompressionMapSpec { key, alphabet }
        })
        .collect();
    CompressionChainSpec { source, maps }
}

fn scalar_opening() -> OpeningClaimsLayout {
    OpeningClaimsLayout::new(4, 1).unwrap()
}

fn standalone(max_opening_log_basis: u32) -> CompressionCatalogContext<'static> {
    CompressionCatalogContext::StandaloneCommitment {
        max_opening_log_basis,
    }
}

fn current_and_opening_specs(lp: &LevelParams) -> Vec<CompressionChainSpec> {
    vec![
        chain_for(
            lp,
            CompressionSourceId::CurrentOuter,
            &lp.b_key,
            &[
                CompressionAlphabet::OpeningBase {
                    log_basis: lp.log_basis,
                },
                CompressionAlphabet::NegativeBinary,
            ],
        ),
        chain_for(
            lp,
            CompressionSourceId::Opening,
            &lp.d_key,
            &[
                CompressionAlphabet::OpeningBase {
                    log_basis: lp.log_basis,
                },
                CompressionAlphabet::NegativeBinary,
            ],
        ),
    ]
}
