use akita_error::AkitaError;

use super::GroupOpenPhaseParams;

/// Nonempty, canonically ordered group storage for one fold.
///
/// The last entry is always the fold's own group. An optional setup prefix may
/// appear only at index zero. The boxed slice prevents callers from changing
/// the collection length after construction.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(super) struct FoldGroups {
    entries: Box<[GroupOpenPhaseParams]>,
}

impl FoldGroups {
    pub(super) fn singleton(own: GroupOpenPhaseParams) -> Self {
        Self {
            entries: Box::new([own]),
        }
    }

    /// Build checked fold-group storage from canonical-order entries.
    pub(super) fn try_from_vec(entries: Vec<GroupOpenPhaseParams>) -> Result<Self, AkitaError> {
        let groups = Self {
            entries: entries.into_boxed_slice(),
        };
        groups.validate_topology()?;
        Ok(groups)
    }

    /// All groups in transcript order.
    #[must_use]
    pub(super) fn as_slice(&self) -> &[GroupOpenPhaseParams] {
        &self.entries
    }

    pub(super) fn validate_topology(&self) -> Result<(), AkitaError> {
        let Some((own, preceding)) = self.entries.split_last() else {
            return Err(AkitaError::InvalidSetup(
                "a fold must contain its own group".into(),
            ));
        };
        if own.setup_natural_len.is_some() {
            return Err(AkitaError::InvalidSetup(
                "a fold's own group cannot be a setup prefix".into(),
            ));
        }
        if preceding
            .iter()
            .enumerate()
            .any(|(index, group)| index != 0 && group.setup_natural_len.is_some())
        {
            return Err(AkitaError::InvalidSetup(
                "a setup prefix may appear only as the first group".into(),
            ));
        }
        Ok(())
    }

    pub(super) fn own(&self) -> &GroupOpenPhaseParams {
        // `singleton` and `try_from_vec` are the only constructors and both
        // establish nonemptiness.
        &self.entries[self.entries.len() - 1]
    }

    pub(super) fn own_mut(&mut self) -> &mut GroupOpenPhaseParams {
        let own_index = self.entries.len() - 1;
        &mut self.entries[own_index]
    }

    pub(super) fn preceding(&self) -> &[GroupOpenPhaseParams] {
        &self.entries[..self.entries.len() - 1]
    }

    pub(super) fn preceding_group(&self, index: usize) -> Option<&GroupOpenPhaseParams> {
        self.preceding().get(index)
    }

    #[cfg(test)]
    pub(super) fn preceding_group_mut(
        &mut self,
        index: usize,
    ) -> Option<&mut GroupOpenPhaseParams> {
        let own_index = self.entries.len() - 1;
        self.entries[..own_index].get_mut(index)
    }

    pub(super) fn setup_prefix(&self) -> Option<&GroupOpenPhaseParams> {
        self.preceding()
            .first()
            .filter(|group| group.setup_natural_len.is_some())
    }

    pub(super) fn precommitted(&self) -> &[GroupOpenPhaseParams] {
        let preceding = self.preceding();
        if self.setup_prefix().is_some() {
            &preceding[1..]
        } else {
            preceding
        }
    }

    pub(super) fn replace_precommitted(
        &mut self,
        groups: Vec<GroupOpenPhaseParams>,
    ) -> Result<(), AkitaError> {
        if groups.iter().any(|group| group.setup_natural_len.is_some()) {
            return Err(AkitaError::InvalidSetup(
                "precommitted groups cannot carry setup-prefix metadata".into(),
            ));
        }
        let prefix = self.setup_prefix().copied();
        let own = *self.own();
        *self = Self::try_from_vec(
            prefix
                .into_iter()
                .chain(groups)
                .chain(std::iter::once(own))
                .collect(),
        )?;
        Ok(())
    }

    pub(super) fn insert_precommitted(
        &mut self,
        group: GroupOpenPhaseParams,
    ) -> Result<(), AkitaError> {
        if group.setup_natural_len.is_some() {
            return Err(AkitaError::InvalidSetup(
                "precommitted groups cannot carry setup-prefix metadata".into(),
            ));
        }
        let mut entries = self.entries.to_vec();
        entries.insert(entries.len() - 1, group);
        *self = Self::try_from_vec(entries)?;
        Ok(())
    }

    pub(super) fn replace_setup_prefix(
        &mut self,
        prefix: Option<GroupOpenPhaseParams>,
    ) -> Result<(), AkitaError> {
        if prefix.is_some_and(|group| group.setup_natural_len.is_none()) {
            return Err(AkitaError::InvalidSetup(
                "setup prefix is missing its natural length".into(),
            ));
        }
        let mut entries = self.entries.to_vec();
        if self.setup_prefix().is_some() {
            entries.remove(0);
        }
        if let Some(prefix) = prefix {
            entries.insert(0, prefix);
        }
        *self = Self::try_from_vec(entries)?;
        Ok(())
    }
}
