//! Checked commitment-compression chain geometry.

use akita_field::{AkitaError, CanonicalField};
use akita_serialization::DEFAULT_MAX_SEQUENCE_LEN;

use crate::dispatch::{protocol_dispatch_tier, slot_dim_supported_for_tier, ProtocolDispatchSlot};
use crate::sis::{
    num_digits_for_bound, rounded_up_collision_inf_norm, sis_table_key_for_linf_bound,
    AjtaiKeyParams, SisModulusFamily, SisTableKey, DEFAULT_SIS_SECURITY_BITS,
};
use crate::{LevelParams, OpeningClaimsLayout, RingRole, MAX_SETUP_MATRIX_FIELD_ELEMENTS};

pub(in crate::layout) mod semantics;

#[allow(dead_code)] // Wired into schedule replay in the compression cutover slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CompressionAlphabet {
    NegativeBinary,
    OpeningBase { log_basis: u32 },
}

#[allow(dead_code)] // Wired into schedule replay in the compression cutover slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionSourceId {
    CurrentOuter,
    PrecommittedOuter { index: usize },
    Opening,
}

#[allow(dead_code)] // Wired into schedule replay in the compression cutover slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CompressionCatalogContext<'a> {
    CoGeneratedLevel { opening: &'a OpeningClaimsLayout },
    StandaloneCommitment { max_opening_log_basis: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CompressionCatalogPurpose {
    CoGenerated {
        relation_layout: crate::layout::relation::RelationLayout,
    },
    Standalone {
        max_opening_log_basis: u32,
        source_key: AjtaiKeyParams,
        field_modulus_minus_one: u128,
    },
}

#[allow(dead_code)] // Wired into schedule replay in the compression cutover slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CompressionMapSpec {
    pub(crate) key: AjtaiKeyParams,
    pub(crate) alphabet: CompressionAlphabet,
}

#[allow(dead_code)] // Wired into schedule replay in the compression cutover slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CompressionChainSpec {
    pub(crate) source: CompressionSourceId,
    pub(crate) maps: Vec<CompressionMapSpec>,
}

#[allow(dead_code)] // Compiled catalog fields are consumed by later protocol slices.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CompiledCompressionMap {
    key: AjtaiKeyParams,
    alphabet: CompressionAlphabet,
    digit_depth: usize,
    input_coeffs: usize,
    output_coeffs: usize,
}

#[allow(dead_code)] // Compiled catalog fields are consumed by later protocol slices.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CompiledCompressionChain {
    source: CompressionSourceId,
    source_output_coeffs: usize,
    maps: Vec<CompiledCompressionMap>,
    payload_coeffs: usize,
}

fn resolve_source_key(
    lp: &LevelParams,
    source: CompressionSourceId,
) -> Result<&AjtaiKeyParams, AkitaError> {
    match source {
        CompressionSourceId::CurrentOuter => Ok(&lp.b_key),
        CompressionSourceId::PrecommittedOuter { index } => lp
            .precommitted_groups
            .get(index)
            .map(|group| &group.b_key)
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "compression precommitted source index is out of range".into(),
                )
            }),
        CompressionSourceId::Opening => Ok(&lp.d_key),
    }
}

#[allow(dead_code)] // Wired into schedule replay in the compression cutover slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ValidatedCompressionCatalog {
    chains: Vec<CompiledCompressionChain>,
    purpose: CompressionCatalogPurpose,
}

impl ValidatedCompressionCatalog {
    #[allow(dead_code)] // Schedule replay consumes this in the next crate cutover.
    pub(crate) fn co_generated_relation_layout(
        &self,
    ) -> Result<&crate::layout::relation::RelationLayout, AkitaError> {
        match &self.purpose {
            CompressionCatalogPurpose::CoGenerated { relation_layout } => Ok(relation_layout),
            CompressionCatalogPurpose::Standalone { .. } => Err(AkitaError::InvalidSetup(
                "standalone compression has no co-generated relation layout".into(),
            )),
        }
    }

    #[allow(dead_code)] // Terminal schedule replay consumes this in the next crate cutover.
    pub(crate) fn terminal_relation_layout<F: CanonicalField>(
        &self,
        lp: &LevelParams,
        opening: &OpeningClaimsLayout,
    ) -> Result<crate::layout::relation::RelationLayout, AkitaError> {
        let (max_opening_log_basis, source_key, field_modulus_minus_one) = match &self.purpose {
            CompressionCatalogPurpose::Standalone {
                max_opening_log_basis,
                source_key,
                field_modulus_minus_one,
            } => (*max_opening_log_basis, source_key, *field_modulus_minus_one),
            CompressionCatalogPurpose::CoGenerated { .. } => {
                return Err(AkitaError::InvalidSetup(
                    "terminal relation requires standalone incoming compression".into(),
                ));
            }
        };
        if (-F::one()).to_canonical_u128() != field_modulus_minus_one {
            return Err(AkitaError::InvalidSetup(
                "terminal compression field identity disagrees with the frozen catalog".into(),
            ));
        }
        if source_key != &lp.b_key || lp.log_basis > max_opening_log_basis {
            return Err(AkitaError::InvalidSetup(
                "terminal compression source key or opening base is incompatible".into(),
            ));
        }
        if !lp.precommitted_groups.is_empty() || self.chains.len() != 1 {
            return Err(AkitaError::InvalidSetup(
                "terminal compression must be exactly the frozen CurrentOuter chain".into(),
            ));
        }
        let chain = self.chains.first().ok_or_else(|| {
            AkitaError::InvalidSetup("terminal compression chain is missing".into())
        })?;
        if chain.source != CompressionSourceId::CurrentOuter {
            return Err(AkitaError::InvalidSetup(
                "terminal compression source must be CurrentOuter".into(),
            ));
        }
        let first_map = chain.maps.first().ok_or_else(|| {
            AkitaError::InvalidSetup("terminal compression first map is missing".into())
        })?;
        if let CompressionAlphabet::OpeningBase { log_basis } = first_map.alphabet {
            if lp.log_basis < log_basis {
                return Err(AkitaError::InvalidSetup(
                    "terminal opening base is smaller than the frozen F-chain base".into(),
                ));
            }
        }
        let local = semantics::compile::<F>(lp, &self.chains)?;
        crate::layout::relation::RelationLayout::compile_terminal_compressed(
            lp,
            opening,
            &local,
            F::modulus_bits(),
        )
    }
}

#[allow(dead_code)] // Reached through the dormant catalog compiler below.
fn checked_coeffs(count: usize, d: usize, label: &str) -> Result<usize, AkitaError> {
    count.checked_mul(d).ok_or_else(|| {
        AkitaError::InvalidSetup(format!("compression {label} coefficient count overflow"))
    })
}

#[allow(dead_code)] // Reached through the dormant catalog compiler below.
fn audit_key(
    key: &AjtaiKeyParams,
    active_family: SisModulusFamily,
    active_security: u16,
) -> Result<SisTableKey, AkitaError> {
    let stored = key.sis_table_key();
    if key.row_len() == 0 || key.col_len() == 0 {
        return Err(AkitaError::InvalidSetup(
            "compression Ajtai key has zero rows or columns".into(),
        ));
    }
    if stored.family != active_family || stored.min_security_bits != active_security {
        return Err(AkitaError::InvalidSetup(
            "compression Ajtai key disagrees with the active SIS family or security floor".into(),
        ));
    }
    let reconstructed = AjtaiKeyParams::try_new(
        stored.min_security_bits,
        stored.family,
        key.row_len(),
        key.col_len(),
        stored.coeff_linf_bound,
        stored.ring_dimension as usize,
    )
    .map_err(|_| {
        AkitaError::InvalidSetup("compression Ajtai key is not canonically SIS certified".into())
    })?;
    if &reconstructed != key {
        return Err(AkitaError::InvalidSetup(
            "compression Ajtai key disagrees with canonical AjtaiKeyParams construction".into(),
        ));
    }
    Ok(stored)
}

#[allow(dead_code)] // Reached through the dormant catalog compiler below.
fn alphabet_facts(
    alphabet: CompressionAlphabet,
    field_bits: u32,
    level_log_basis: u32,
) -> Result<usize, AkitaError> {
    match alphabet {
        CompressionAlphabet::NegativeBinary => Ok(field_bits as usize),
        CompressionAlphabet::OpeningBase { log_basis } => {
            if log_basis == 0 || log_basis >= 128 || log_basis > level_log_basis {
                return Err(AkitaError::InvalidSetup(
                    "compression opening-base log_basis must be in 1..128 and no larger than the level base".into(),
                ));
            }
            Ok(num_digits_for_bound(field_bits, field_bits, log_basis))
        }
    }
}

#[allow(dead_code)] // Reached through the dormant catalog compiler below.
fn compile_chain<F: CanonicalField>(
    lp: &LevelParams,
    context: CompressionCatalogContext<'_>,
    range_log_basis: u32,
    gen_ring_dim: usize,
    active_family: SisModulusFamily,
    spec: &CompressionChainSpec,
) -> Result<CompiledCompressionChain, AkitaError> {
    if !(2..=3).contains(&spec.maps.len()) {
        return Err(AkitaError::InvalidSetup(
            "compression chain depth must be in 2..=3".into(),
        ));
    }
    if gen_ring_dim == 0 {
        return Err(AkitaError::InvalidSetup(
            "compression gen_ring_dim must be non-zero".into(),
        ));
    }

    let source_key = resolve_source_key(lp, spec.source)?;
    let source_table_key = source_key.sis_table_key();
    let source_d = source_table_key.ring_dimension as usize;
    let required_source_bucket = rounded_up_collision_inf_norm(
        DEFAULT_SIS_SECURITY_BITS,
        active_family,
        source_d,
        range_log_basis,
    )
    .ok_or_else(|| {
        AkitaError::InvalidSetup("compression source has no required SIS collision bucket".into())
    })?;
    let audited_source = audit_key(source_key, active_family, DEFAULT_SIS_SECURITY_BITS)?;
    if audited_source.coeff_linf_bound < required_source_bucket {
        return Err(AkitaError::InvalidSetup(
            "compression source key is below its required SIS collision bucket".into(),
        ));
    }
    let tier = protocol_dispatch_tier::<F>();
    let source_role = match spec.source {
        CompressionSourceId::CurrentOuter | CompressionSourceId::PrecommittedOuter { .. } => {
            RingRole::Outer
        }
        CompressionSourceId::Opening => RingRole::Opening,
    };
    if !source_d.is_power_of_two()
        || !slot_dim_supported_for_tier(tier, ProtocolDispatchSlot::Role(source_role), source_d)
    {
        return Err(AkitaError::InvalidSetup(
            "compression source ring dimension is outside its role dispatch policy".into(),
        ));
    }
    if !gen_ring_dim.is_multiple_of(source_d) {
        return Err(AkitaError::InvalidSetup(
            "compression source ring dimension does not divide gen_ring_dim".into(),
        ));
    }
    let source_setup_coeffs = source_key
        .row_len()
        .checked_mul(source_key.col_len())
        .and_then(|count| count.checked_mul(source_d))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("compression source setup footprint overflow".into())
        })?;
    if source_setup_coeffs > MAX_SETUP_MATRIX_FIELD_ELEMENTS {
        return Err(AkitaError::InvalidSetup(
            "compression source setup footprint exceeds cap".into(),
        ));
    }
    let source_output_coeffs = checked_coeffs(source_key.row_len(), source_d, "source output")?;
    if source_output_coeffs > DEFAULT_MAX_SEQUENCE_LEN {
        return Err(AkitaError::InvalidSetup(
            "compression source output exceeds sequence cap".into(),
        ));
    }

    let modulus = crate::field_modulus::<F>();
    let field_bits = 128 - modulus.saturating_sub(1).leading_zeros();
    if field_bits == 0 {
        return Err(AkitaError::InvalidSetup(
            "compression field modulus has zero bit length".into(),
        ));
    }
    let mut previous_output = source_output_coeffs;
    let mut maps = Vec::with_capacity(spec.maps.len());
    for (index, map) in spec.maps.iter().enumerate() {
        if index != 0 && matches!(map.alphabet, CompressionAlphabet::OpeningBase { .. }) {
            return Err(AkitaError::InvalidSetup(
                "compression opening-base alphabet is permitted only on the first map".into(),
            ));
        }
        if let CompressionAlphabet::OpeningBase { log_basis } = map.alphabet {
            match (context, spec.source) {
                (
                    CompressionCatalogContext::CoGeneratedLevel { .. },
                    CompressionSourceId::CurrentOuter | CompressionSourceId::Opening,
                ) if log_basis != lp.log_basis => {
                    return Err(AkitaError::InvalidSetup(
                        "co-generated current/opening first-map base must equal the level base"
                            .into(),
                    ));
                }
                (
                    CompressionCatalogContext::StandaloneCommitment { .. },
                    CompressionSourceId::CurrentOuter,
                ) if log_basis != 4 => {
                    return Err(AkitaError::InvalidSetup(
                        "standalone commitment opening-base first map must use log_basis=4".into(),
                    ));
                }
                _ => {}
            }
        }
        let digit_depth = alphabet_facts(map.alphabet, field_bits, range_log_basis)?;
        let stored = map.key.sis_table_key();
        let required_bucket = match map.alphabet {
            CompressionAlphabet::NegativeBinary => {
                let key = sis_table_key_for_linf_bound(
                    DEFAULT_SIS_SECURITY_BITS,
                    active_family,
                    stored.ring_dimension,
                    1,
                )
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "negative-binary map has no exact SIS table row".into(),
                    )
                })?;
                key.coeff_linf_bound
            }
            CompressionAlphabet::OpeningBase { .. } => rounded_up_collision_inf_norm(
                DEFAULT_SIS_SECURITY_BITS,
                active_family,
                stored.ring_dimension as usize,
                range_log_basis,
            )
            .ok_or_else(|| {
                AkitaError::InvalidSetup("opening-base map has no required SIS table row".into())
            })?,
        };
        let table_key = audit_key(&map.key, active_family, DEFAULT_SIS_SECURITY_BITS)?;
        match map.alphabet {
            CompressionAlphabet::NegativeBinary
                if table_key.coeff_linf_bound != required_bucket =>
            {
                return Err(AkitaError::InvalidSetup(
                    "negative-binary map must use its exact SIS collision bucket".into(),
                ));
            }
            CompressionAlphabet::OpeningBase { .. }
                if table_key.coeff_linf_bound < required_bucket =>
            {
                return Err(AkitaError::InvalidSetup(
                    "opening-base map is below its required SIS collision bucket".into(),
                ));
            }
            _ => {}
        }
        let d = table_key.ring_dimension as usize;
        if !d.is_power_of_two()
            || !slot_dim_supported_for_tier(tier, ProtocolDispatchSlot::Compression, d)
            || !slot_dim_supported_for_tier(tier, ProtocolDispatchSlot::Ntt, d)
        {
            return Err(AkitaError::InvalidSetup(format!(
                "compression map {index} ring dimension {d} is outside dispatch policy"
            )));
        }
        if !gen_ring_dim.is_multiple_of(d) {
            return Err(AkitaError::InvalidSetup(format!(
                "compression map {index} ring dimension {d} does not divide gen_ring_dim={gen_ring_dim}"
            )));
        }
        let expected_input = previous_output.checked_mul(digit_depth).ok_or_else(|| {
            AkitaError::InvalidSetup("compression decomposed input length overflow".into())
        })?;
        let actual_input = checked_coeffs(map.key.col_len(), d, "map input")?;
        if actual_input > DEFAULT_MAX_SEQUENCE_LEN {
            return Err(AkitaError::InvalidSetup(format!(
                "compression map {index} input exceeds sequence cap"
            )));
        }
        if actual_input != expected_input {
            return Err(AkitaError::InvalidSetup(format!(
                "compression map {index} input has {actual_input} coefficients, expected {expected_input}"
            )));
        }
        let output_coeffs = checked_coeffs(map.key.row_len(), d, "map output")?;
        if output_coeffs > DEFAULT_MAX_SEQUENCE_LEN {
            return Err(AkitaError::InvalidSetup(format!(
                "compression map {index} output exceeds sequence cap"
            )));
        }
        let setup_coeffs = map
            .key
            .row_len()
            .checked_mul(map.key.col_len())
            .and_then(|count| count.checked_mul(d))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("compression setup footprint overflow".into())
            })?;
        if setup_coeffs > MAX_SETUP_MATRIX_FIELD_ELEMENTS {
            return Err(AkitaError::InvalidSetup(format!(
                "compression map {index} setup footprint {setup_coeffs} exceeds {MAX_SETUP_MATRIX_FIELD_ELEMENTS}"
            )));
        }
        maps.push(CompiledCompressionMap {
            key: map.key.clone(),
            alphabet: map.alphabet,
            digit_depth,
            input_coeffs: actual_input,
            output_coeffs,
        });
        previous_output = output_coeffs;
    }
    if previous_output > DEFAULT_MAX_SEQUENCE_LEN {
        return Err(AkitaError::InvalidSetup(format!(
            "compression payload coefficient count {previous_output} exceeds {DEFAULT_MAX_SEQUENCE_LEN}"
        )));
    }
    Ok(CompiledCompressionChain {
        source: spec.source,
        source_output_coeffs,
        maps,
        payload_coeffs: previous_output,
    })
}

/// Validate and compile the complete compression catalog for one level.
#[allow(dead_code)] // Wired into schedule replay in the compression cutover slice.
pub(crate) fn validate_and_compile<F: CanonicalField>(
    lp: &LevelParams,
    context: CompressionCatalogContext<'_>,
    gen_ring_dim: usize,
    specs: Vec<CompressionChainSpec>,
) -> Result<ValidatedCompressionCatalog, AkitaError> {
    if lp.log_basis == 0 || lp.log_basis >= 128 {
        return Err(AkitaError::InvalidSetup(
            "compression level log_basis must be in 1..128".into(),
        ));
    }
    let (range_log_basis, co_generated_opening) = match context {
        CompressionCatalogContext::CoGeneratedLevel { opening } => {
            opening.check()?;
            lp.validate_root_opening_batch(opening)?;
            (lp.log_basis, Some(opening))
        }
        CompressionCatalogContext::StandaloneCommitment {
            max_opening_log_basis,
        } => {
            if !lp.precommitted_groups.is_empty() {
                return Err(AkitaError::InvalidSetup(
                    "standalone compression cannot include precommitted groups".into(),
                ));
            }
            if !(4..128).contains(&max_opening_log_basis) || max_opening_log_basis < lp.log_basis {
                return Err(AkitaError::InvalidSetup(
                    "standalone max_opening_log_basis must be in 4..128 and cover lp.log_basis"
                        .into(),
                ));
            }
            (max_opening_log_basis, None)
        }
    };
    let precommitted_count = match context {
        CompressionCatalogContext::CoGeneratedLevel { .. } => lp.precommitted_groups.len(),
        CompressionCatalogContext::StandaloneCommitment { .. } => 0,
    };
    let opening_count = usize::from(matches!(
        context,
        CompressionCatalogContext::CoGeneratedLevel { .. }
    ));
    let expected_source_count = 1usize
        .checked_add(precommitted_count)
        .and_then(|count| count.checked_add(opening_count))
        .ok_or_else(|| AkitaError::InvalidSetup("compression source count overflow".into()))?;
    if expected_source_count > DEFAULT_MAX_SEQUENCE_LEN {
        return Err(AkitaError::InvalidSetup(format!(
            "compression source count {expected_source_count} exceeds cap {DEFAULT_MAX_SEQUENCE_LEN}"
        )));
    }
    let mut expected_sources = Vec::with_capacity(expected_source_count);
    expected_sources.push(CompressionSourceId::CurrentOuter);
    expected_sources.extend(
        (0..precommitted_count).map(|index| CompressionSourceId::PrecommittedOuter { index }),
    );
    if opening_count == 1 {
        expected_sources.push(CompressionSourceId::Opening);
    }
    if specs.len() != expected_source_count
        || !specs
            .iter()
            .zip(&expected_sources)
            .all(|(spec, expected)| spec.source == *expected)
    {
        return Err(AkitaError::InvalidSetup(format!(
            "compression catalog sources must be exactly {expected_sources:?} in canonical order"
        )));
    }
    let active_family = match protocol_dispatch_tier::<F>() {
        crate::ProtocolRingDispatchTierId::Fp128 => SisModulusFamily::Q128,
        crate::ProtocolRingDispatchTierId::Fp64 => SisModulusFamily::Q64,
        crate::ProtocolRingDispatchTierId::Fp32 => SisModulusFamily::Q32,
    };
    let chains = specs
        .iter()
        .map(|spec| {
            compile_chain::<F>(
                lp,
                context,
                range_log_basis,
                gen_ring_dim,
                active_family,
                spec,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let purpose = if let Some(opening) = co_generated_opening {
        let local = semantics::compile::<F>(lp, &chains)?;
        CompressionCatalogPurpose::CoGenerated {
            relation_layout: crate::layout::relation::RelationLayout::compile_compressed(
                lp,
                opening,
                &local,
                F::modulus_bits(),
            )?,
        }
    } else {
        CompressionCatalogPurpose::Standalone {
            max_opening_log_basis: range_log_basis,
            source_key: lp.b_key.clone(),
            field_modulus_minus_one: (-F::one()).to_canonical_u128(),
        }
    };
    Ok(ValidatedCompressionCatalog { chains, purpose })
}

#[cfg(test)]
mod tests {
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

    #[test]
    fn compiles_whole_catalog_and_derives_geometry() {
        let lp = level();
        let catalog = validate_and_compile::<Prime128OffsetA7F7>(
            &lp,
            CompressionCatalogContext::CoGeneratedLevel {
                opening: &scalar_opening(),
            },
            64,
            current_and_opening_specs(&lp),
        )
        .unwrap();
        assert_eq!(catalog.chains.len(), 2);
        assert_eq!(catalog.chains[0].source, CompressionSourceId::CurrentOuter);
        assert_eq!(catalog.chains[1].source, CompressionSourceId::Opening);
        assert_eq!(catalog.chains[0].maps.len(), 2);
        assert_eq!(catalog.chains[0].maps[0].digit_depth, 22);
        assert_eq!(catalog.chains[0].maps[1].digit_depth, 128);
        assert_eq!(
            catalog.chains[0].maps[0].key.sis_table_key().ring_dimension as usize,
            D
        );
        assert_eq!(
            catalog.chains[0].maps[0].input_coeffs,
            catalog.chains[0].source_output_coeffs * 22
        );
        assert_eq!(
            catalog.chains[0].payload_coeffs,
            catalog.chains[0].maps[1].output_coeffs
        );
    }

    #[test]
    fn resolves_current_precommitted_and_opening_sources_in_canonical_order() {
        let mut lp = level();
        let pre_b = key(SisModulusFamily::Q128, D, 63, 2);
        let group = PolynomialGroupLayout::new(3, 1);
        lp.precommitted_groups.push(PrecommittedLevelParams {
            layout: PrecommittedGroupParams::from_params(group, &lp),
            a_key: lp.a_key.clone(),
            b_key: pre_b.clone(),
            num_blocks: 1,
            block_len: 1,
            num_digits_commit: 1,
            num_digits_open: 1,
            num_digits_fold_one: 1,
        });
        let opening =
            OpeningClaimsLayout::from_root_groups(&[group], PolynomialGroupLayout::new(4, 1))
                .unwrap();
        let specs = vec![
            chain_for(
                &lp,
                CompressionSourceId::CurrentOuter,
                &lp.b_key,
                &[CompressionAlphabet::NegativeBinary; 2],
            ),
            chain_for(
                &lp,
                CompressionSourceId::PrecommittedOuter { index: 0 },
                &pre_b,
                &[
                    CompressionAlphabet::OpeningBase { log_basis: 4 },
                    CompressionAlphabet::NegativeBinary,
                ],
            ),
            chain_for(
                &lp,
                CompressionSourceId::Opening,
                &lp.d_key,
                &[CompressionAlphabet::NegativeBinary; 2],
            ),
        ];
        let catalog = validate_and_compile::<Prime128OffsetA7F7>(
            &lp,
            CompressionCatalogContext::CoGeneratedLevel { opening: &opening },
            64,
            specs,
        )
        .unwrap();
        assert_eq!(catalog.chains.len(), 3);
        assert_eq!(
            catalog.chains[1].source,
            CompressionSourceId::PrecommittedOuter { index: 0 }
        );
        let standalone_spec = chain_for(
            &lp,
            CompressionSourceId::CurrentOuter,
            &lp.b_key,
            &[CompressionAlphabet::NegativeBinary; 2],
        );
        assert!(validate_and_compile::<Prime128OffsetA7F7>(
            &lp,
            standalone(lp.log_basis),
            64,
            vec![standalone_spec],
        )
        .is_err());
    }

    #[test]
    fn rejects_missing_duplicate_out_of_order_and_wrong_purpose_sources() {
        let lp = level();
        let opening = scalar_opening();
        let good = current_and_opening_specs(&lp);
        for specs in [
            vec![good[0].clone()],
            vec![good[0].clone(), good[0].clone()],
            vec![good[1].clone(), good[0].clone()],
            vec![
                good[0].clone(),
                CompressionChainSpec {
                    source: CompressionSourceId::PrecommittedOuter { index: 0 },
                    maps: good[1].maps.clone(),
                },
            ],
        ] {
            assert!(validate_and_compile::<Prime128OffsetA7F7>(
                &lp,
                CompressionCatalogContext::CoGeneratedLevel { opening: &opening },
                64,
                specs,
            )
            .is_err());
        }
        assert!(validate_and_compile::<Prime128OffsetA7F7>(
            &lp,
            standalone(lp.log_basis),
            64,
            good,
        )
        .is_err());
        let standalone_negative_binary = chain_for(
            &lp,
            CompressionSourceId::CurrentOuter,
            &lp.b_key,
            &[CompressionAlphabet::NegativeBinary; 2],
        );
        assert!(validate_and_compile::<Prime128OffsetA7F7>(
            &lp,
            standalone(lp.log_basis - 1),
            64,
            vec![standalone_negative_binary],
        )
        .is_err());
    }

    #[test]
    fn purpose_enforces_base_and_source_semantics() {
        let lp = level();
        let mut wrong_current_base = current_and_opening_specs(&lp);
        wrong_current_base[0].maps[0].alphabet = CompressionAlphabet::OpeningBase { log_basis: 4 };
        assert!(validate_and_compile::<Prime128OffsetA7F7>(
            &lp,
            CompressionCatalogContext::CoGeneratedLevel {
                opening: &scalar_opening(),
            },
            64,
            wrong_current_base,
        )
        .is_err());

        let mut wrong_opening_base = current_and_opening_specs(&lp);
        wrong_opening_base[1].maps[0].alphabet = CompressionAlphabet::OpeningBase { log_basis: 4 };
        assert!(validate_and_compile::<Prime128OffsetA7F7>(
            &lp,
            CompressionCatalogContext::CoGeneratedLevel {
                opening: &scalar_opening(),
            },
            64,
            wrong_opening_base,
        )
        .is_err());

        let standalone_base_five = chain_for(
            &lp,
            CompressionSourceId::CurrentOuter,
            &lp.b_key,
            &[
                CompressionAlphabet::OpeningBase { log_basis: 5 },
                CompressionAlphabet::NegativeBinary,
            ],
        );
        assert!(validate_and_compile::<Prime128OffsetA7F7>(
            &lp,
            standalone(lp.log_basis),
            64,
            vec![standalone_base_five],
        )
        .is_err());
    }

    #[test]
    fn rejects_chain_depths_outside_two_or_three() {
        let lp = level();
        for depth in [0, 1, 4] {
            let mut chain = chain_for(
                &lp,
                CompressionSourceId::CurrentOuter,
                &lp.b_key,
                &[CompressionAlphabet::NegativeBinary; 2],
            );
            chain.maps.resize(depth, chain.maps[0].clone());
            assert!(validate_and_compile::<Prime128OffsetA7F7>(
                &lp,
                standalone(lp.log_basis),
                64,
                vec![chain],
            )
            .is_err());
        }
    }

    #[test]
    fn accepts_depth_three_with_negative_binary_later_maps() {
        let lp = level();
        let spec = chain_for(
            &lp,
            CompressionSourceId::CurrentOuter,
            &lp.b_key,
            &[
                CompressionAlphabet::OpeningBase { log_basis: 4 },
                CompressionAlphabet::NegativeBinary,
                CompressionAlphabet::NegativeBinary,
            ],
        );
        let catalog = validate_and_compile::<Prime128OffsetA7F7>(
            &lp,
            standalone(lp.log_basis),
            64,
            vec![spec],
        )
        .unwrap();
        assert_eq!(catalog.chains[0].maps.len(), 3);
    }

    #[test]
    fn rejects_noncanonical_or_insecure_map_keys() {
        let lp = level();
        let base = chain_for(
            &lp,
            CompressionSourceId::CurrentOuter,
            &lp.b_key,
            &[CompressionAlphabet::NegativeBinary; 2],
        );
        let original = &base.maps[0].key;
        let stored = original.sis_table_key();
        let mutations = [
            AjtaiKeyParams::new_unchecked(
                138,
                SisModulusFamily::Q64,
                original.row_len(),
                original.col_len(),
                stored.coeff_linf_bound,
                D,
            ),
            AjtaiKeyParams::new_unchecked(
                137,
                SisModulusFamily::Q128,
                original.row_len(),
                original.col_len(),
                stored.coeff_linf_bound,
                D,
            ),
            AjtaiKeyParams::new_unchecked(
                138,
                SisModulusFamily::Q128,
                original.row_len(),
                original.col_len(),
                3,
                D,
            ),
            AjtaiKeyParams::new_unchecked(
                138,
                SisModulusFamily::Q128,
                0,
                original.col_len(),
                stored.coeff_linf_bound,
                D,
            ),
            AjtaiKeyParams::new_unchecked(
                138,
                SisModulusFamily::Q128,
                original.row_len(),
                usize::MAX,
                stored.coeff_linf_bound,
                D,
            ),
        ];
        for mutation in mutations {
            let mut spec = base.clone();
            spec.maps[0].key = mutation;
            assert!(validate_and_compile::<Prime128OffsetA7F7>(
                &lp,
                standalone(lp.log_basis),
                64,
                vec![spec],
            )
            .is_err());
        }
    }

    #[test]
    fn rejects_source_key_outside_field_family_and_production_security() {
        let mut lp = level();
        lp.b_key = key(SisModulusFamily::Q64, D, 63, 1);
        let spec = chain_for(
            &lp,
            CompressionSourceId::CurrentOuter,
            &lp.b_key,
            &[CompressionAlphabet::NegativeBinary; 2],
        );
        assert!(validate_and_compile::<Prime128OffsetA7F7>(
            &lp,
            standalone(lp.log_basis),
            64,
            vec![spec],
        )
        .is_err());

        let mut lp = level();
        let source = &lp.b_key;
        lp.b_key = AjtaiKeyParams::new_unchecked(
            DEFAULT_SIS_SECURITY_BITS - 1,
            SisModulusFamily::Q128,
            source.row_len(),
            source.col_len(),
            source.coeff_linf_bound(),
            D,
        );
        let spec = chain_for(
            &level(),
            CompressionSourceId::CurrentOuter,
            &lp.b_key,
            &[CompressionAlphabet::NegativeBinary; 2],
        );
        assert!(validate_and_compile::<Prime128OffsetA7F7>(
            &lp,
            standalone(lp.log_basis),
            64,
            vec![spec],
        )
        .is_err());
    }

    #[test]
    fn source_uses_recomputed_level_collision_bound() {
        let mut too_small = level();
        too_small.b_key = key(SisModulusFamily::Q128, D, 31, 1);
        let spec = chain_for(
            &too_small,
            CompressionSourceId::CurrentOuter,
            &too_small.b_key,
            &[CompressionAlphabet::NegativeBinary; 2],
        );
        assert!(validate_and_compile::<Prime128OffsetA7F7>(
            &too_small,
            standalone(too_small.log_basis),
            64,
            vec![spec],
        )
        .is_err());

        let mut conservative = level();
        conservative.b_key = key(SisModulusFamily::Q128, D, 127, 1);
        let spec = chain_for(
            &conservative,
            CompressionSourceId::CurrentOuter,
            &conservative.b_key,
            &[CompressionAlphabet::NegativeBinary; 2],
        );
        assert!(validate_and_compile::<Prime128OffsetA7F7>(
            &conservative,
            standalone(conservative.log_basis),
            64,
            vec![spec],
        )
        .is_ok());
    }

    #[test]
    fn rejects_unsupported_sis_dimensions_dispatch_and_gen_divisibility() {
        let lp = level();
        let base = chain_for(
            &lp,
            CompressionSourceId::CurrentOuter,
            &lp.b_key,
            &[CompressionAlphabet::NegativeBinary; 2],
        );
        for d in [8, 16] {
            let mut spec = base.clone();
            let original = &spec.maps[0].key;
            spec.maps[0].key = AjtaiKeyParams::new_unchecked(
                138,
                SisModulusFamily::Q128,
                original.row_len(),
                original.col_len(),
                2,
                d,
            );
            assert!(validate_and_compile::<Prime128OffsetA7F7>(
                &lp,
                standalone(lp.log_basis),
                64,
                vec![spec],
            )
            .is_err());
        }
        let mut dispatch_rejected = base.clone();
        let expected_input = lp.b_key.row_len() * D * 128;
        dispatch_rejected.maps[0].key = key(SisModulusFamily::Q128, 128, 1, expected_input / 128);
        assert!(validate_and_compile::<Prime128OffsetA7F7>(
            &lp,
            standalone(lp.log_basis),
            128,
            vec![dispatch_rejected],
        )
        .is_err());
        assert!(validate_and_compile::<Prime128OffsetA7F7>(
            &lp,
            standalone(lp.log_basis),
            48,
            vec![base],
        )
        .is_err());
    }

    #[test]
    fn rejects_invalid_bases_and_chain_discontinuity() {
        let mut lp = level();
        for log_basis in [0, 7, 128] {
            let spec = chain_for(
                &lp,
                CompressionSourceId::CurrentOuter,
                &lp.b_key,
                &[CompressionAlphabet::NegativeBinary; 2],
            );
            let mut spec = spec;
            spec.maps[0].alphabet = CompressionAlphabet::OpeningBase { log_basis };
            assert!(validate_and_compile::<Prime128OffsetA7F7>(
                &lp,
                standalone(lp.log_basis),
                64,
                vec![spec],
            )
            .is_err());
        }
        let mut later_opening_base = chain_for(
            &lp,
            CompressionSourceId::CurrentOuter,
            &lp.b_key,
            &[CompressionAlphabet::NegativeBinary; 2],
        );
        later_opening_base.maps[1].alphabet = CompressionAlphabet::OpeningBase { log_basis: 4 };
        assert!(validate_and_compile::<Prime128OffsetA7F7>(
            &lp,
            standalone(lp.log_basis),
            64,
            vec![later_opening_base],
        )
        .is_err());
        for invalid_level_base in [0, 128] {
            lp.log_basis = invalid_level_base;
            let spec = chain_for(
                &level(),
                CompressionSourceId::CurrentOuter,
                &lp.b_key,
                &[CompressionAlphabet::NegativeBinary; 2],
            );
            assert!(validate_and_compile::<Prime128OffsetA7F7>(
                &lp,
                standalone(lp.log_basis),
                64,
                vec![spec],
            )
            .is_err());
        }

        let lp = level();
        let mut spec = chain_for(
            &lp,
            CompressionSourceId::CurrentOuter,
            &lp.b_key,
            &[CompressionAlphabet::NegativeBinary; 2],
        );
        let key = &spec.maps[1].key;
        spec.maps[1].key = AjtaiKeyParams::new_unchecked(
            key.min_security_bits(),
            key.sis_family(),
            key.row_len(),
            key.col_len() + 1,
            key.coeff_linf_bound(),
            D,
        );
        assert!(validate_and_compile::<Prime128OffsetA7F7>(
            &lp,
            standalone(lp.log_basis),
            64,
            vec![spec],
        )
        .is_err());
    }

    #[test]
    fn opening_base_accepts_larger_canonical_bucket() {
        let lp = level();
        let mut spec = chain_for(
            &lp,
            CompressionSourceId::CurrentOuter,
            &lp.b_key,
            &[
                CompressionAlphabet::OpeningBase { log_basis: 4 },
                CompressionAlphabet::NegativeBinary,
            ],
        );
        let first_col_len = spec.maps[0].key.col_len();
        spec.maps[0].key = key(SisModulusFamily::Q128, D, 127, first_col_len);
        let second_col_len = spec.maps[0].key.row_len() * 128;
        spec.maps[1].key = key(SisModulusFamily::Q128, D, 1, second_col_len);
        let catalog = validate_and_compile::<Prime128OffsetA7F7>(
            &lp,
            standalone(lp.log_basis),
            64,
            vec![spec],
        )
        .unwrap();
        assert_eq!(catalog.chains[0].maps[0].key.coeff_linf_bound(), 127);
    }

    #[test]
    fn frozen_standalone_terminal_join_binds_key_base_and_field() {
        use crate::layout::relation::{
            RelationGroupId, RelationRowId, RelationRowInputs, RelationRowRhs,
        };

        let lp = level();
        let spec = chain_for(
            &lp,
            CompressionSourceId::CurrentOuter,
            &lp.b_key,
            &[
                CompressionAlphabet::OpeningBase { log_basis: 4 },
                CompressionAlphabet::NegativeBinary,
            ],
        );
        let catalog =
            validate_and_compile::<Prime128OffsetA7F7>(&lp, standalone(6), 64, vec![spec]).unwrap();
        let opening = scalar_opening();
        let layout = catalog
            .terminal_relation_layout::<Prime128OffsetA7F7>(&lp, &opening)
            .unwrap();
        let b = layout
            .row_plan()
            .family(RelationRowId::B {
                group: RelationGroupId::Current,
            })
            .unwrap();
        assert_eq!(b.rhs(), RelationRowRhs::Zero);
        let RelationRowInputs::B {
            compression_input: Some(input),
            ..
        } = b.inputs()
        else {
            panic!("terminal B must carry frozen F input");
        };
        assert_eq!(input.log_basis(), 4);
        assert!(layout.row_plan().family(RelationRowId::D).is_err());
        assert!(layout.row_plan().families().iter().all(|family| !matches!(
            family.id(),
            RelationRowId::Compression {
                source: CompressionSourceId::Opening,
                ..
            }
        )));

        let mut too_small_base = lp.clone();
        too_small_base.log_basis = 3;
        assert!(catalog
            .terminal_relation_layout::<Prime128OffsetA7F7>(&too_small_base, &opening)
            .is_err());
        let mut wrong_key = lp.clone();
        wrong_key.b_key = key(SisModulusFamily::Q128, D, 63, 2);
        assert!(catalog
            .terminal_relation_layout::<Prime128OffsetA7F7>(&wrong_key, &opening)
            .is_err());
        assert!(catalog
            .terminal_relation_layout::<Prime64Offset59>(&lp, &opening)
            .is_err());
    }

    #[test]
    fn binary_first_terminal_join_has_no_opening_base_lower_bound() {
        let lp = level();
        let spec = chain_for(
            &lp,
            CompressionSourceId::CurrentOuter,
            &lp.b_key,
            &[
                CompressionAlphabet::NegativeBinary,
                CompressionAlphabet::NegativeBinary,
            ],
        );
        let catalog =
            validate_and_compile::<Prime128OffsetA7F7>(&lp, standalone(6), 64, vec![spec]).unwrap();
        let mut terminal = lp.clone();
        terminal.log_basis = 2;
        assert!(catalog
            .terminal_relation_layout::<Prime128OffsetA7F7>(&terminal, &scalar_opening())
            .is_ok());
    }

    #[test]
    fn standalone_range_envelope_prices_conservative_base() {
        let mut lp = level();
        lp.log_basis = 2;
        let opening_base = chain_for_profile(
            CompressionSourceId::CurrentOuter,
            &lp.b_key,
            &[
                CompressionAlphabet::OpeningBase { log_basis: 4 },
                CompressionAlphabet::NegativeBinary,
            ],
            SisModulusFamily::Q128,
            128,
            D,
            6,
        );
        let catalog = validate_and_compile::<Prime128OffsetA7F7>(
            &lp,
            standalone(6),
            64,
            vec![opening_base.clone()],
        )
        .unwrap();
        assert!(matches!(
            catalog.purpose,
            CompressionCatalogPurpose::Standalone {
                max_opening_log_basis: 6,
                ..
            }
        ));
        assert_eq!(catalog.chains[0].maps[0].key.coeff_linf_bound(), 63);

        let mut underpriced = opening_base;
        let col_len = underpriced.maps[0].key.col_len();
        underpriced.maps[0].key = key(SisModulusFamily::Q128, D, 3, col_len);
        assert!(validate_and_compile::<Prime128OffsetA7F7>(
            &lp,
            standalone(6),
            64,
            vec![underpriced],
        )
        .is_err());

        let negative_binary = chain_for_profile(
            CompressionSourceId::CurrentOuter,
            &lp.b_key,
            &[CompressionAlphabet::NegativeBinary; 2],
            SisModulusFamily::Q128,
            128,
            D,
            6,
        );
        assert!(validate_and_compile::<Prime128OffsetA7F7>(
            &lp,
            standalone(6),
            64,
            vec![negative_binary],
        )
        .is_ok());
    }

    #[test]
    fn reduced_field_profiles_use_native_negative_binary_depths() {
        let mut q64 = LevelParams::params_only(
            SisModulusFamily::Q64,
            64,
            6,
            1,
            1,
            1,
            SparseChallengeConfig::pm1_only(64),
        );
        q64.b_key = key(SisModulusFamily::Q64, 32, 63, 1);
        q64.d_key = key(SisModulusFamily::Q64, 32, 63, 1);
        let q64_spec = chain_for_profile(
            CompressionSourceId::CurrentOuter,
            &q64.b_key,
            &[CompressionAlphabet::NegativeBinary; 2],
            SisModulusFamily::Q64,
            64,
            32,
            6,
        );
        let q64_catalog =
            validate_and_compile::<Prime64Offset59>(&q64, standalone(6), 64, vec![q64_spec])
                .unwrap();
        assert_eq!(q64_catalog.chains[0].maps[0].digit_depth, 64);

        let mut q32 = LevelParams::params_only(
            SisModulusFamily::Q32,
            64,
            6,
            1,
            1,
            1,
            SparseChallengeConfig::pm1_only(64),
        );
        q32.b_key = key(SisModulusFamily::Q32, 64, 63, 1);
        q32.d_key = key(SisModulusFamily::Q32, 64, 63, 1);
        let q32_spec = chain_for_profile(
            CompressionSourceId::CurrentOuter,
            &q32.b_key,
            &[CompressionAlphabet::NegativeBinary; 2],
            SisModulusFamily::Q32,
            32,
            32,
            6,
        );
        let q32_catalog =
            validate_and_compile::<Prime32Offset99>(&q32, standalone(6), 64, vec![q32_spec])
                .unwrap();
        assert_eq!(q32_catalog.chains[0].maps[0].digit_depth, 32);

        let q128 = level();
        let cross_family = chain_for(
            &q128,
            CompressionSourceId::CurrentOuter,
            &q128.b_key,
            &[CompressionAlphabet::NegativeBinary; 2],
        );
        assert!(validate_and_compile::<Prime64Offset59>(
            &q128,
            standalone(6),
            64,
            vec![cross_family],
        )
        .is_err());
    }
}
