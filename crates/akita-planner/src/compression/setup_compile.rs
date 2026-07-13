//! Compile aggregated compression setup facts into global planner objectives.

use akita_field::AkitaError;
use akita_types::{
    aggregate_catalog_projections, AggregatedCompressionSetup, CompressionCatalogProjection,
    CompressionMapHintShape, SetupMatrixEnvelope,
};

use super::global_setup_objectives;

/// Aggregated compression hints plus global setup/cache objectives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressionSetupArtifacts {
    gen_ring_dim: usize,
    aggregated: AggregatedCompressionSetup,
    global_setup_prefix_coeffs: usize,
    global_cache_field_coeffs: usize,
}

impl CompressionSetupArtifacts {
    #[must_use]
    pub fn gen_ring_dim(&self) -> usize {
        self.gen_ring_dim
    }

    #[must_use]
    pub fn map_hints(&self) -> &[CompressionMapHintShape] {
        self.aggregated.map_hints()
    }

    #[must_use]
    pub fn ntt_requirements(&self) -> &[(usize, usize)] {
        self.aggregated.ntt_requirements()
    }

    #[must_use]
    pub fn max_flat_setup_prefix_coeffs(&self) -> usize {
        self.aggregated.max_flat_setup_prefix_coeffs()
    }

    #[must_use]
    pub fn coalesced_cache_field_coeffs(&self) -> usize {
        self.aggregated.coalesced_cache_field_coeffs()
    }

    #[must_use]
    pub fn global_setup_prefix_coeffs(&self) -> usize {
        self.global_setup_prefix_coeffs
    }

    #[must_use]
    pub fn global_cache_field_coeffs(&self) -> usize {
        self.global_cache_field_coeffs
    }

    /// Whole generation rings required by the rounded global setup prefix.
    #[must_use]
    pub fn global_setup_prefix_rings(&self) -> usize {
        self.global_setup_prefix_coeffs / self.gen_ring_dim
    }

    /// Inflate a setup envelope to cover the rounded compression setup prefix.
    pub fn inflate_setup_envelope(
        &self,
        envelope: &mut SetupMatrixEnvelope,
    ) -> Result<(), AkitaError> {
        envelope.max_setup_len = envelope.max_setup_len.max(self.global_setup_prefix_rings());
        Ok(())
    }
}

/// Compile checked catalog projections into one compression setup envelope.
///
/// Generation dimension is taken from the projections themselves. Every
/// projection must share one `gen_ring_dim`; an empty projection list is
/// rejected because there is then no validated dimension to bind.
pub fn compile_compression_setup_artifacts(
    base_envelope: SetupMatrixEnvelope,
    projections: &[&CompressionCatalogProjection],
) -> Result<CompressionSetupArtifacts, AkitaError> {
    if projections.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "compression setup compilation requires at least one catalog projection".into(),
        ));
    }
    if base_envelope.max_setup_len == 0 {
        return Err(AkitaError::InvalidSetup(
            "compression setup base envelope must be non-zero".into(),
        ));
    }
    let aggregated = aggregate_catalog_projections(projections)?;
    let gen_ring_dim = aggregated.gen_ring_dim().ok_or_else(|| {
        AkitaError::InvalidSetup("aggregated compression setup is missing gen_ring_dim".into())
    })?;
    let base_setup_coeffs = base_envelope
        .max_setup_len
        .checked_mul(gen_ring_dim)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("compression base setup coefficient count overflow".into())
        })?;
    let (global_setup_prefix_coeffs, global_cache_field_coeffs) = global_setup_objectives(
        base_setup_coeffs,
        gen_ring_dim,
        aggregated.max_flat_setup_prefix_coeffs(),
        aggregated.ntt_requirements(),
    )?;
    Ok(CompressionSetupArtifacts {
        gen_ring_dim,
        aggregated,
        global_setup_prefix_coeffs,
        global_cache_field_coeffs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::Prime128OffsetA7F7;
    use akita_types::{
        validate_compression_catalog, AjtaiKeyParams, CompressionAlphabet,
        CompressionCatalogContext, CompressionChainSpec, CompressionMapSpec, CompressionSourceId,
        LevelParams, SisModulusFamily, DEFAULT_SIS_SECURITY_BITS,
    };

    type F = Prime128OffsetA7F7;

    fn key(d: usize, raw_bound: u128, col_len: usize) -> AjtaiKeyParams {
        let table_key = akita_types::sis::sis_table_key_for_linf_bound(
            DEFAULT_SIS_SECURITY_BITS,
            SisModulusFamily::Q128,
            d as u32,
            raw_bound,
        )
        .expect("test SIS row");
        AjtaiKeyParams::try_new_with_min_rank(table_key, col_len).expect("test secure key")
    }

    fn projection(map_d: usize, terminal_rows: usize) -> CompressionCatalogProjection {
        let mut lp = LevelParams::params_only(
            SisModulusFamily::Q128,
            64,
            4,
            1,
            1,
            1,
            SparseChallengeConfig::pm1_only(64),
        )
        .with_decomp(1, 1, 1, 1, 0)
        .unwrap();
        lp.b_key = key(32, 63, 1);
        lp.stamp_role_dims_from_keys();
        let source_d = lp.b_key.sis_table_key().ring_dimension as usize;
        let mut previous_output = lp.b_key.row_len() * source_d;
        let maps = [CompressionAlphabet::NegativeBinary; 2]
            .into_iter()
            .enumerate()
            .map(|(map_index, alphabet)| {
                let input_coeffs = previous_output * 128;
                assert_eq!(input_coeffs % map_d, 0);
                let mut map_key = key(map_d, 1, input_coeffs / map_d);
                if map_index == 1 && terminal_rows != 0 {
                    map_key = AjtaiKeyParams::try_new(
                        DEFAULT_SIS_SECURITY_BITS,
                        SisModulusFamily::Q128,
                        map_key.row_len() + terminal_rows,
                        map_key.col_len(),
                        1,
                        map_d,
                    )
                    .expect("terminal key");
                }
                previous_output = map_key.row_len() * map_d;
                CompressionMapSpec::new(map_key, alphabet)
            })
            .collect();
        validate_compression_catalog::<F>(
            &lp,
            CompressionCatalogContext::StandaloneCommitment,
            64,
            vec![CompressionChainSpec::new(
                CompressionSourceId::CurrentOuter,
                4,
                maps,
            )],
        )
        .expect("catalog")
        .project_for_schedule()
        .expect("projection")
    }

    #[test]
    fn compile_matches_single_catalog_global_objectives() {
        let base_envelope = SetupMatrixEnvelope { max_setup_len: 1 };
        let projection = projection(32, 0);
        let artifacts =
            compile_compression_setup_artifacts(base_envelope, &[&projection]).expect("artifacts");
        assert_eq!(artifacts.gen_ring_dim(), projection.gen_ring_dim());
        let base_setup_coeffs = base_envelope
            .max_setup_len
            .checked_mul(projection.gen_ring_dim())
            .expect("base coeffs");
        let expected = super::global_setup_objectives(
            base_setup_coeffs,
            projection.gen_ring_dim(),
            projection.max_flat_setup_prefix_coeffs(),
            projection.ntt_requirements(),
        )
        .expect("objectives");
        assert_eq!(
            (
                artifacts.global_setup_prefix_coeffs(),
                artifacts.global_cache_field_coeffs()
            ),
            expected
        );
        assert_eq!(artifacts.map_hints(), projection.map_hints());
        assert_eq!(
            artifacts.max_flat_setup_prefix_coeffs(),
            projection.max_flat_setup_prefix_coeffs()
        );
        assert_eq!(
            artifacts.coalesced_cache_field_coeffs(),
            projection.coalesced_cache_field_coeffs()
        );
    }

    #[test]
    fn compile_uses_aggregated_max_prefix_across_catalogs() {
        let base_envelope = SetupMatrixEnvelope { max_setup_len: 2 };
        let left = projection(32, 0);
        let right = projection(64, 1);
        let artifacts = compile_compression_setup_artifacts(base_envelope, &[&left, &right])
            .expect("artifacts");
        let aggregated = aggregate_catalog_projections(&[&left, &right]).expect("aggregate");
        let base_setup_coeffs = base_envelope
            .max_setup_len
            .checked_mul(left.gen_ring_dim())
            .expect("base coeffs");
        let expected = super::global_setup_objectives(
            base_setup_coeffs,
            left.gen_ring_dim(),
            aggregated.max_flat_setup_prefix_coeffs(),
            aggregated.ntt_requirements(),
        )
        .expect("objectives");
        assert_eq!(
            (
                artifacts.global_setup_prefix_coeffs(),
                artifacts.global_cache_field_coeffs()
            ),
            expected
        );
        assert!(
            aggregated.max_flat_setup_prefix_coeffs()
                >= left
                    .max_flat_setup_prefix_coeffs()
                    .max(right.max_flat_setup_prefix_coeffs())
        );
    }

    #[test]
    fn compile_rejects_empty_projection_list() {
        let base_envelope = SetupMatrixEnvelope { max_setup_len: 1 };
        assert!(compile_compression_setup_artifacts(base_envelope, &[]).is_err());
    }

    #[test]
    fn inflate_uses_rounded_global_prefix_rings() {
        let base_envelope = SetupMatrixEnvelope { max_setup_len: 1 };
        let projection = projection(32, 0);
        let artifacts =
            compile_compression_setup_artifacts(base_envelope, &[&projection]).expect("artifacts");
        let mut envelope = SetupMatrixEnvelope { max_setup_len: 1 };
        artifacts
            .inflate_setup_envelope(&mut envelope)
            .expect("inflate");
        assert_eq!(
            envelope.max_setup_len,
            artifacts.global_setup_prefix_rings()
        );
    }
}
