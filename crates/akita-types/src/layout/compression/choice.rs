use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionAlphabet {
    NegativeBinary,
    OpeningBase { log_basis: u32 },
}

pub fn compression_digit_depth(
    alphabet: CompressionAlphabet,
    field_bits: u32,
    max_opening_log_basis: u32,
) -> Result<usize, AkitaError> {
    match alphabet {
        CompressionAlphabet::NegativeBinary => Ok(field_bits as usize),
        CompressionAlphabet::OpeningBase { log_basis } => {
            if log_basis == 0 || log_basis >= 128 || log_basis > max_opening_log_basis {
                return Err(AkitaError::InvalidSetup(
                    "compression opening-base log_basis must be in 1..128 and within its frozen envelope".into(),
                ));
            }
            Ok(num_digits_for_bound(field_bits, field_bits, log_basis))
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionSourceId {
    CurrentOuter,
    PrecommittedOuter { index: usize },
    Opening,
}

/// One compression map reconstructed from a compact schedule choice.
#[allow(dead_code)] // Wired into schedule replay in the compression cutover slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressionMapSpec {
    pub(super) key: AjtaiKeyParams,
    pub(super) alphabet: CompressionAlphabet,
}

impl CompressionMapSpec {
    #[must_use]
    pub fn new(key: AjtaiKeyParams, alphabet: CompressionAlphabet) -> Self {
        Self { key, alphabet }
    }
}

/// One source-assigned compression chain reconstructed for catalog validation.
#[allow(dead_code)] // Wired into schedule replay in the compression cutover slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressionChainSpec {
    pub(super) source: CompressionSourceId,
    pub(super) max_opening_log_basis: u32,
    pub(super) maps: Vec<CompressionMapSpec>,
}

impl CompressionChainSpec {
    #[must_use]
    pub fn new(
        source: CompressionSourceId,
        max_opening_log_basis: u32,
        maps: Vec<CompressionMapSpec>,
    ) -> Self {
        Self {
            source,
            max_opening_log_basis,
            maps,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionCatalogContext<'a> {
    CoGeneratedLevel { opening: &'a OpeningClaimsLayout },
    StandaloneCommitment,
    TerminalFold { opening: &'a OpeningClaimsLayout },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompressionMapChoice {
    pub ring_d: u32,
    pub alphabet: CompressionAlphabet,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionChainChoice {
    Two([CompressionMapChoice; 2]),
    Three([CompressionMapChoice; 3]),
}

impl CompressionChainChoice {
    #[must_use]
    pub const fn maps(&self) -> &[CompressionMapChoice] {
        match self {
            Self::Two(maps) => maps,
            Self::Three(maps) => maps,
        }
    }
}

/// One F chain together with the opening-base envelope used for SIS pricing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrozenCompressionChainChoice {
    /// Digest of the exact B key whose output this F chain compresses.
    pub source_key_digest: [u8; 32],
    pub max_opening_log_basis: u32,
    pub chain: CompressionChainChoice,
}

impl FrozenCompressionChainChoice {
    #[must_use]
    pub fn new(
        source_key: &AjtaiKeyParams,
        max_opening_log_basis: u32,
        chain: CompressionChainChoice,
    ) -> Self {
        Self {
            source_key_digest: source_key.compression_source_descriptor_digest(),
            max_opening_log_basis,
            chain,
        }
    }
}

/// Source-assigned F choices shared by standalone creation and later use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompressionFChoice<'a> {
    pub current_outer: FrozenCompressionChainChoice,
    pub precommitted_outer: &'a [FrozenCompressionChainChoice],
}

impl CompressionFChoice<'_> {
    pub fn descriptor_bytes(&self) -> Result<Vec<u8>, AkitaError> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"AKITA-COMPRESSION-F-CHOICE-V1");
        append_frozen_chain(&mut bytes, &self.current_outer)?;
        crate::descriptor_bytes::push_usize(&mut bytes, self.precommitted_outer.len());
        for chain in self.precommitted_outer {
            append_frozen_chain(&mut bytes, chain)?;
        }
        ensure_projection_descriptor_len(&bytes)?;
        Ok(bytes)
    }

    pub fn descriptor_digest(&self) -> Result<[u8; 32], AkitaError> {
        digest_bytes(self.descriptor_bytes()?)
    }
}

/// Complete compact choice: shared F choices plus an optional level-local H.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompressionChoice<'a> {
    pub f: CompressionFChoice<'a>,
    pub opening: Option<CompressionChainChoice>,
}

impl CompressionChoice<'_> {
    pub fn descriptor_bytes(&self) -> Result<Vec<u8>, AkitaError> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"AKITA-COMPRESSION-CHOICE-V2");
        bytes.extend_from_slice(&self.f.descriptor_bytes()?);
        match self.opening {
            Some(opening) => {
                bytes.push(1);
                append_choice_chain(&mut bytes, &opening)?;
            }
            None => bytes.push(0),
        }
        ensure_projection_descriptor_len(&bytes)?;
        Ok(bytes)
    }

    pub fn descriptor_digest(&self) -> Result<[u8; 32], AkitaError> {
        digest_bytes(self.descriptor_bytes()?)
    }

    pub fn replay<F: CanonicalField>(
        &self,
        lp: &LevelParams,
        context: CompressionCatalogContext<'_>,
    ) -> Result<ValidatedCompressionCatalog, AkitaError> {
        let mut sources = Vec::new();
        match context {
            CompressionCatalogContext::CoGeneratedLevel { .. } => {
                let opening = self.opening.ok_or_else(|| {
                    AkitaError::InvalidSetup("co-generated compression choice is missing H".into())
                })?;
                require_precommitted_count(self.f.precommitted_outer, lp)?;
                sources.extend(f_sources(self.f).map(|(source, frozen)| {
                    (
                        source,
                        frozen.max_opening_log_basis,
                        frozen.chain,
                        Some(frozen.source_key_digest),
                    )
                }));
                sources.push((CompressionSourceId::Opening, lp.log_basis, opening, None));
            }
            CompressionCatalogContext::StandaloneCommitment => {
                if self.opening.is_some() || !self.f.precommitted_outer.is_empty() {
                    return Err(AkitaError::InvalidSetup(
                        "standalone compression requires exactly the current F slot".into(),
                    ));
                }
                let frozen = self.f.current_outer;
                sources.push((
                    CompressionSourceId::CurrentOuter,
                    frozen.max_opening_log_basis,
                    frozen.chain,
                    Some(frozen.source_key_digest),
                ));
            }
            CompressionCatalogContext::TerminalFold { .. } => {
                if self.opening.is_some() {
                    return Err(AkitaError::InvalidSetup(
                        "terminal compression choice must not contain H".into(),
                    ));
                }
                require_precommitted_count(self.f.precommitted_outer, lp)?;
                sources.extend(f_sources(self.f).map(|(source, frozen)| {
                    (
                        source,
                        frozen.max_opening_log_basis,
                        frozen.chain,
                        Some(frozen.source_key_digest),
                    )
                }));
            }
        }

        let active_family = match protocol_dispatch_tier::<F>() {
            crate::ProtocolRingDispatchTierId::Fp128 => SisModulusFamily::Q128,
            crate::ProtocolRingDispatchTierId::Fp64 => SisModulusFamily::Q64,
            crate::ProtocolRingDispatchTierId::Fp32 => SisModulusFamily::Q32,
        };
        let specs = sources
            .into_iter()
            .map(|(source, max, chain, expected_key_digest)| {
                replay_chain::<F>(lp, active_family, source, max, chain, expected_key_digest)
            })
            .collect::<Result<Vec<_>, _>>()?;
        validate_compression_catalog::<F>(lp, context, lp.ring_dimension, specs)
    }
}

fn require_precommitted_count(
    choices: &[FrozenCompressionChainChoice],
    lp: &LevelParams,
) -> Result<(), AkitaError> {
    if choices.len() != lp.precommitted_groups.len() {
        return Err(AkitaError::InvalidSetup(
            "compression precommitted F slots disagree with authenticated groups".into(),
        ));
    }
    Ok(())
}

fn f_sources(
    f: CompressionFChoice<'_>,
) -> impl Iterator<Item = (CompressionSourceId, FrozenCompressionChainChoice)> + '_ {
    std::iter::once((CompressionSourceId::CurrentOuter, f.current_outer)).chain(
        f.precommitted_outer
            .iter()
            .copied()
            .enumerate()
            .map(|(index, chain)| (CompressionSourceId::PrecommittedOuter { index }, chain)),
    )
}

fn replay_chain<F: CanonicalField>(
    lp: &LevelParams,
    family: SisModulusFamily,
    source: CompressionSourceId,
    max_opening_log_basis: u32,
    chain: CompressionChainChoice,
    expected_key_digest: Option<[u8; 32]>,
) -> Result<CompressionChainSpec, AkitaError> {
    let source_key = resolve_source_key(lp, source)?;
    if expected_key_digest
        .is_some_and(|expected| expected != source_key.compression_source_descriptor_digest())
    {
        return Err(AkitaError::InvalidSetup(
            "compression source key disagrees with its frozen descriptor".into(),
        ));
    }
    let mut previous_output = source_key
        .row_len()
        .checked_mul(source_key.sis_table_key().ring_dimension as usize)
        .ok_or_else(|| AkitaError::InvalidSetup("compression source size overflow".into()))?;
    let mut maps = Vec::with_capacity(chain.maps().len());
    for map in chain.maps() {
        let d = map.ring_d as usize;
        let digit_depth =
            compression_digit_depth(map.alphabet, F::modulus_bits(), max_opening_log_basis)?;
        let input_coeffs = previous_output.checked_mul(digit_depth).ok_or_else(|| {
            AkitaError::InvalidSetup("compression replay input size overflow".into())
        })?;
        if d == 0 || !input_coeffs.is_multiple_of(d) {
            return Err(AkitaError::InvalidSetup(
                "compression replay input is not divisible by its native dimension".into(),
            ));
        }
        let table_key = match map.alphabet {
            CompressionAlphabet::NegativeBinary => {
                sis_table_key_for_linf_bound(DEFAULT_SIS_SECURITY_BITS, family, map.ring_d, 1)
            }
            CompressionAlphabet::OpeningBase { .. } => rounded_up_collision_inf_norm(
                DEFAULT_SIS_SECURITY_BITS,
                family,
                d,
                max_opening_log_basis,
            )
            .map(|coeff_linf_bound| SisTableKey {
                min_security_bits: DEFAULT_SIS_SECURITY_BITS,
                family,
                ring_dimension: map.ring_d,
                coeff_linf_bound,
            }),
        }
        .ok_or_else(|| {
            AkitaError::InvalidSetup(
                "compression replay map is absent from the shipped SIS tables".into(),
            )
        })?;
        let key = AjtaiKeyParams::try_new_with_min_rank(table_key, input_coeffs / d)?;
        previous_output = key.row_len().checked_mul(d).ok_or_else(|| {
            AkitaError::InvalidSetup("compression replay output size overflow".into())
        })?;
        maps.push(CompressionMapSpec::new(key, map.alphabet));
    }
    Ok(CompressionChainSpec::new(
        source,
        max_opening_log_basis,
        maps,
    ))
}

fn append_frozen_chain(
    bytes: &mut Vec<u8>,
    frozen: &FrozenCompressionChainChoice,
) -> Result<(), AkitaError> {
    bytes.extend_from_slice(&frozen.source_key_digest);
    crate::descriptor_bytes::push_u32(bytes, frozen.max_opening_log_basis);
    append_choice_chain(bytes, &frozen.chain)
}

fn append_choice_chain(
    bytes: &mut Vec<u8>,
    chain: &CompressionChainChoice,
) -> Result<(), AkitaError> {
    bytes.push(match chain {
        CompressionChainChoice::Two(_) => 2,
        CompressionChainChoice::Three(_) => 3,
    });
    for map in chain.maps() {
        crate::descriptor_bytes::push_u32(bytes, map.ring_d);
        match map.alphabet {
            CompressionAlphabet::NegativeBinary => bytes.push(0),
            CompressionAlphabet::OpeningBase { log_basis } => {
                bytes.push(1);
                crate::descriptor_bytes::push_u32(bytes, log_basis);
            }
        }
        ensure_projection_descriptor_len(bytes)?;
    }
    Ok(())
}

fn digest_bytes(bytes: Vec<u8>) -> Result<[u8; 32], AkitaError> {
    ensure_projection_descriptor_len(&bytes)?;
    Ok(crate::descriptor_bytes::blake2b_256(&bytes))
}
