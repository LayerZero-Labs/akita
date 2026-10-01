use super::*;

pub(crate) fn append_sparse_challenge_descriptor_bytes(
    bytes: &mut Vec<u8>,
    config: &SparseChallengeConfig,
) {
    bytes.push(0);
    push_usize(bytes, config.count_pm1);
    push_usize(bytes, config.count_pm2);
}

impl CommittedGroupParams {
    /// Canonical byte encoding used to order semantically distinct level candidates.
    ///
    /// This is an ordering descriptor, not a wire encoding or transcript commitment.
    #[must_use]
    pub fn canonical_descriptor_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        self.append_descriptor_bytes(&mut bytes);
        bytes
    }

    /// Checked wire geometry for this level's final-group B image.
    pub fn outer_payload_geometry(&self) -> Result<crate::CommitmentPayloadGeometry, AkitaError> {
        let logical_rows = self
            .outer_slice_count()
            .logical_output_rows(self.outer().matrix.output_rank())?;
        crate::CommitmentPayloadGeometry::for_mode(
            self.payload_mode,
            self.outer().matrix.sis_modulus_profile(),
            logical_rows,
            self.role_dims().d_b(),
        )
    }

    /// Checked wire geometry for this level's shared D image.
    pub fn opening_payload_geometry(&self) -> Result<crate::CommitmentPayloadGeometry, AkitaError> {
        crate::CommitmentPayloadGeometry::for_mode(
            self.payload_mode,
            self.open().matrix.sis_modulus_profile(),
            self.open().matrix.output_rank(),
            self.role_dims().d_d(),
        )
    }

    /// Whether every B/D image at this level fits the compression policy cap.
    pub fn compression_sources_supported(&self) -> Result<bool, AkitaError> {
        if !self.payload_mode.is_compressed() {
            return Ok(true);
        }
        let final_outer = self.outer_slice_count().complete_source_coefficients(
            self.outer().matrix.output_rank(),
            self.role_dims().d_b(),
        )?;
        if crate::CompressionChainPlan::try_for_complete_source(
            self.outer().matrix.sis_modulus_profile(),
            final_outer,
        )?
        .is_none()
        {
            return Ok(false);
        }
        for group in self.preceding_group_iter() {
            let source = group
                .profile
                .outer_slice_count
                .complete_source_coefficients(
                    group.profile.outer.matrix.output_rank(),
                    group.profile.outer.matrix.ring_dimension(),
                )?;
            if crate::CompressionChainPlan::try_for_complete_source(
                group.profile.outer.matrix.sis_modulus_profile(),
                source,
            )?
            .is_none()
            {
                return Ok(false);
            }
        }
        let opening = self
            .open()
            .matrix
            .output_rank()
            .checked_mul(self.role_dims().d_d())
            .ok_or_else(|| AkitaError::InvalidSetup("D compression shape overflow".into()))?;
        Ok(crate::CompressionChainPlan::try_for_complete_source(
            self.open().matrix.sis_modulus_profile(),
            opening,
        )?
        .is_some())
    }

    /// Append the descriptor digest encoding for this parameter set.
    ///
    /// Kept next to [`CommittedGroupParams`] so protocol-affecting field changes are
    /// reviewed with their Fiat-Shamir binding.
    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        self.append_descriptor_bytes_with_payload_mode(bytes, self.payload_mode);
    }

    pub(crate) fn append_descriptor_bytes_with_payload_mode(
        &self,
        bytes: &mut Vec<u8>,
        payload_mode: crate::CommitmentPayloadMode,
    ) {
        bytes.push(payload_mode.tag());
        bytes.push(self.ring_relation_mode.tag());
        self.source_encoding.append_descriptor_bytes(bytes);
        self.opening_method().append_descriptor_bytes(bytes);
        push_u32(bytes, self.inner().digits.log_basis);
        push_u32(bytes, self.outer().digits.log_basis);
        push_u32(bytes, self.open().digits.log_basis);
        self.inner().matrix.append_descriptor_bytes(bytes);
        self.outer().matrix.append_descriptor_bytes(bytes);
        self.open().matrix.append_descriptor_bytes(bytes);
        push_usize(bytes, self.blocks().live_ring_elements_per_claim);
        push_usize(bytes, self.blocks().positions_per_block);
        push_usize(bytes, self.blocks().live_blocks);
        self.outer_slice_count().append_descriptor_bytes(bytes);
        append_sparse_challenge_descriptor_bytes(bytes, &self.fold_challenge_config());
        push_usize(bytes, self.inner().digits.num_digits);
        push_usize(bytes, self.outer().digits.num_digits);
        push_usize(bytes, self.open().digits.num_digits);
        push_usize(bytes, self.num_digits_fold());
        // Chunk binding is appended only when the level is chunked, so
        // single-chunk descriptors stay byte-for-byte identical to the historical
        // layout (the flag-off no-op invariant). When chunked, bind the chunk
        // count and activated-level count into the Fiat-Shamir digest.
        if self.witness_chunk.num_chunks != 1 {
            self.witness_chunk.append_descriptor_bytes(bytes);
        }

        if !self.precommitted_groups().is_empty() {
            push_usize(bytes, self.precommitted_groups().len());
            for group in self.precommitted_groups() {
                group.append_descriptor_bytes(bytes);
            }
        }
        if let Some(setup_prefix) = self.setup_prefix() {
            bytes.push(1);
            setup_prefix.append_setup_prefix_descriptor_bytes(bytes);
        } else {
            bytes.push(0);
        }
    }
}
