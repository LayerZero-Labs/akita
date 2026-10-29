use std::cell::RefCell;

use super::*;

/// Inputs shared by every split considered for one recursive level.
///
/// A search is prepared for one [`RecursiveCandidateRequest`], and every
/// split walk over it uses that request.
pub(super) struct RecursiveLevelSearch {
    pub(super) num_chunks: usize,
    pub(super) num_ring_elems: usize,
    pub(super) reduced_vars: usize,
    pub(super) current_witness_len: usize,
    pub(super) setup_prefixes: Vec<Option<akita_params::GroupOpenPhaseParams>>,
    /// What the modeled walk visited at each split, for the universal walk.
    split_visits: RefCell<Vec<RecordedSplit>>,
}

type SplitVisit = (LayoutCandidateScore, CommittedGroupParams, usize);

/// One split's candidates in visit order, and the core they were built from.
struct RecordedSplit {
    /// The opening of the request whose walk recorded the split. A search
    /// serves one request, and the selective-ℓ2 path builds its own request
    /// over the same search without walking it; keying the opening keeps a
    /// walk from another request from replaying these visits.
    opening: PlannerOpeningCandidate,
    split: usize,
    relation_domain: RelationSearchDomain,
    core: RecursiveCandidateCore,
    visits: Vec<SplitVisit>,
}

impl RecursiveLevelSearch {
    /// The visits recorded for `split`, if they were built from `core`.
    ///
    /// Everything a walk derives after a split's core is a function of the
    /// core, the split, the relation domain, the request and this search.
    /// The modeled and universal walks of one request differ only in the
    /// source moment, which enters only the core (its modeled ℓ∞ cap), so a
    /// universal walk whose core equals the modeled one at a split visits
    /// exactly what the modeled walk visited there.
    pub(super) fn recorded_split_visits(
        &self,
        opening: PlannerOpeningCandidate,
        split: usize,
        relation_domain: RelationSearchDomain,
        core: &RecursiveCandidateCore,
    ) -> Option<Vec<SplitVisit>> {
        self.split_visits
            .borrow()
            .iter()
            .find(|recorded| {
                recorded.opening == opening
                    && recorded.split == split
                    && recorded.relation_domain == relation_domain
                    && recorded.core == *core
            })
            .map(|recorded| recorded.visits.clone())
    }

    pub(super) fn record_split_visits(
        &self,
        opening: PlannerOpeningCandidate,
        split: usize,
        relation_domain: RelationSearchDomain,
        core: RecursiveCandidateCore,
        visits: Vec<SplitVisit>,
    ) {
        let mut recorded = self.split_visits.borrow_mut();
        if !recorded.iter().any(|existing| {
            existing.opening == opening
                && existing.split == split
                && existing.relation_domain == relation_domain
                && existing.core == core
        }) {
            recorded.push(RecordedSplit {
                opening,
                split,
                relation_domain,
                core,
                visits,
            });
        }
    }
}

pub(super) fn prepare_recursive_level_search(
    request: &RecursiveCandidateRequest<'_>,
    setup_prefix: RecursiveSetupPrefix<'_>,
) -> Result<Option<RecursiveLevelSearch>, AkitaError> {
    let RecursiveCandidateRequest {
        policy,
        opening,
        dimensions,
        current_witness_len,
        log_basis_open,
        fold_level,
        ..
    } = *request;
    let num_chunks = policy.chunks_at_level(fold_level);
    dimensions.validate_role_projection()?;
    opening.validate_for(policy.claim_ext_degree, dimensions)?;
    let d_a = dimensions.d_a();
    if current_witness_len == 0 {
        return Ok(None);
    }
    // The previous fold owns a compact field-coefficient buffer. It need not
    // end on the next A-ring boundary; commitment alignment pads only the
    // transient ring view. Plan from the live coefficient count, rounding up
    // solely to determine the next fold's block geometry.
    let num_ring_elems = current_witness_len.div_ceil(d_a);
    let reduced_vars = num_ring_elems
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("recursive witness capacity overflow".to_string()))?
        .max(1)
        .trailing_zeros() as usize;

    if reduced_vars <= 2 || reduced_vars >= 53 {
        return Err(AkitaError::InvalidSetup(format!(
            "recursive fold candidate reduced_vars={reduced_vars} is outside \
             the optimizable range [3, 52]"
        )));
    }

    let setup_prefixes = match setup_prefix {
        RecursiveSetupPrefix::Search { cache, natural_len } => {
            let n_prefix = padded_setup_prefix_len(natural_len);
            let groups = derive_setup_prefix_groups(
                cache,
                SetupPrefixSearchRequest {
                    policy,
                    opening,
                    log_basis_open,
                    n_prefix,
                    num_chunks,
                    inner_ring_dimension: d_a,
                    outer_ring_dimension: dimensions.d_b(),
                    guide: request.guide.and_then(|guide| guide.setup_prefix),
                    masking: request.masking,
                },
            )?;
            if groups.is_empty() {
                return Ok(None);
            }
            groups
                .into_iter()
                .map(|group| Some(akita_params::scheduled_setup_prefix(natural_len, group)))
                .collect()
        }
        RecursiveSetupPrefix::None => vec![None],
    };
    Ok(Some(RecursiveLevelSearch {
        num_chunks,
        num_ring_elems,
        reduced_vars,
        current_witness_len,
        setup_prefixes,
        split_visits: RefCell::new(Vec::new()),
    }))
}

pub(super) fn attach_recursive_setup_prefix(
    setup_prefix: Option<&akita_params::GroupOpenPhaseParams>,
    extension_degree: usize,
    hiding: bool,
    mut candidate_params: CommittedGroupParams,
) -> Result<CommittedGroupParams, AkitaError> {
    candidate_params.set_setup_prefix(setup_prefix.cloned())?;
    if let Some(prefix) = &candidate_params.setup_prefix() {
        let prefix_d_width =
            prefix.d_segment_width(extension_degree, candidate_params.role_dims().d_d())?;
        let total_d_width = candidate_params
            .open()
            .matrix
            .input_width()
            .checked_add(prefix_d_width)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("setup-prefix shared D width overflow".to_string())
            })?;
        if !hiding {
            candidate_params.open_matrix = OpenCommitMatrixParams::try_new_with_min_rank(
                candidate_params.open().matrix.sis_table_key(),
                total_d_width,
            )?;
        } else {
            // Hiding randomness stays after every group's columns and is
            // resized for the rank the widened D matrix selects.
            let key = candidate_params.open().matrix.sis_table_key();
            let Some((open_matrix, randomness)) = super::super::price_with_hiding_randomness(
                true,
                total_d_width - candidate_params.open_randomness_columns,
                candidate_params.role_dims().d_d(),
                key.modulus_profile,
                candidate_params.open().digits.log_basis,
                |width| OpenCommitMatrixParams::try_new_with_min_rank(key, width).ok(),
                OpenCommitMatrixParams::output_rank,
            )?
            else {
                return Err(AkitaError::UnsupportedSchedule(
                    "setup-prefix D matrix has no hiding fixed point".into(),
                ));
            };
            candidate_params.open_matrix = open_matrix;
            candidate_params.open_randomness_columns = randomness;
        }
    }
    Ok(candidate_params)
}

pub(super) fn finalize_recursive_level_candidate(
    policy: &PlannerPolicy,
    search: &RecursiveLevelSearch,
    candidate_params: CommittedGroupParams,
) -> Result<Option<(LayoutCandidateScore, CommittedGroupParams, usize)>, AkitaError> {
    let Some(next_witness_len) = planned_next_witness_len(
        policy.decomposition.field_bits(),
        policy.claim_ext_degree,
        &candidate_params,
        1,
        search.num_chunks,
    )?
    else {
        return Ok(None);
    };
    let score = layout_candidate_score(
        next_witness_len,
        candidate_params.blocks().live_blocks,
        search.num_chunks,
    )?;
    Ok(Some((score, candidate_params, next_witness_len)))
}
