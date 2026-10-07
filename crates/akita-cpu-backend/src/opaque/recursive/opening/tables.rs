use super::*;

/// Dense EOR group with one transparent factor shared by all witness members.
#[derive(Debug, Clone)]
pub struct ExtensionOpeningReductionGroup<E: Field> {
    pub(in crate::opaque::recursive::opening) terms: Vec<ExtensionOpeningReductionTerm<E>>,
    pub(in crate::opaque::recursive::opening) factor: Vec<E>,
    extra_point: Vec<E>,
    extra_round: usize,
    extra_factor_eval: E,
}

impl<E: Field> ExtensionOpeningReductionGroup<E> {
    /// Construct a group whose members share one transparent factor table.
    ///
    /// # Errors
    ///
    /// Returns an error if there are no members or any witness/factor table is
    /// malformed.
    pub fn new(
        terms: Vec<ExtensionOpeningReductionTerm<E>>,
        factor_evals: Vec<E>,
    ) -> Result<Self, AkitaError> {
        if terms.is_empty() {
            return Err(AkitaError::InvalidInput(
                "extension-opening reduction group requires at least one term".to_string(),
            ));
        }
        for term in &terms {
            validate_reduction_tables(&term.witness, &factor_evals)?;
        }
        Ok(Self {
            terms,
            factor: factor_evals,
            extra_point: Vec::new(),
            extra_round: 0,
            extra_factor_eval: E::one(),
        })
    }

    /// Extend this group over additional high variables without copying its
    /// witness or factor tables.
    ///
    /// # Errors
    ///
    /// Returns an error if the combined virtual table length overflows.
    pub fn extend_cylindrically(mut self, extra_point: Vec<E>) -> Result<Self, AkitaError> {
        let native_rounds = num_rounds_from_table_len(self.factor.len())?;
        let total_rounds = native_rounds
            .checked_add(extra_point.len())
            .ok_or_else(|| {
                AkitaError::InvalidInput(
                    "extension-opening cylindrical domain overflow".to_string(),
                )
            })?;
        reduction_table_len(total_rounds)?;
        self.extra_point = extra_point;
        Ok(self)
    }

    /// Current Boolean-domain table length, including virtual high variables.
    pub(crate) fn domain_len(&self) -> usize {
        self.factor
            .len()
            .checked_shl(
                u32::try_from(self.extra_point.len().saturating_sub(self.extra_round))
                    .unwrap_or(u32::MAX),
            )
            .unwrap_or(0)
    }

    /// Number of witness members sharing this group's factor.
    pub(crate) fn num_terms(&self) -> usize {
        self.terms.len()
    }

    pub(in crate::opaque::recursive::opening) fn final_terms(&self) -> Option<Vec<(E, E, E)>> {
        if self.factor.len() != 1
            || self.extra_round != self.extra_point.len()
            || self.terms.iter().any(|term| term.witness.len() != 1)
        {
            return None;
        }
        let factor = self.factor[0] * self.extra_factor_eval;
        Some(
            self.terms
                .iter()
                .map(|term| (term.coeff, term.witness[0], factor))
                .collect(),
        )
    }
}

impl<E: Field + Unreduced + Fold> ExtensionOpeningReductionGroup<E> {
    pub(in crate::opaque::recursive::opening) fn accumulate_into(
        &mut self,
        constant: &mut E,
        quadratic: &mut E,
    ) {
        if self.factor.len() > 1 {
            for term in &mut self.terms {
                match term.cached_accumulate.take() {
                    Some((cached_constant, cached_quadratic)) => {
                        *constant += cached_constant;
                        *quadratic += cached_quadratic;
                    }
                    None => {
                        let (round_constant, round_quadratic) =
                            accumulate_dense_round(&term.witness, &self.factor, term.coeff);
                        *constant += round_constant;
                        *quadratic += round_quadratic;
                    }
                }
            }
            return;
        }

        if let Some(&point) = self.extra_point.get(self.extra_round) {
            let factor = self.factor[0] * self.extra_factor_eval * (E::one() - point);
            for term in &self.terms {
                *constant += term.coeff * term.witness[0] * factor;
            }
        }
    }

    pub(in crate::opaque::recursive::opening) fn ingest_challenge(&mut self, r_round: E) {
        if self.domain_len() <= 1 {
            return;
        }
        if self.factor.len() > 1 {
            let previous_len = self.factor.len();
            match self.terms.split_first_mut() {
                Some((first, remaining)) if previous_len >= 4 => {
                    let (constant, quadratic) = fused_fold_group_head_and_accumulate(
                        &mut first.witness,
                        &mut self.factor,
                        r_round,
                    );
                    first.cached_accumulate =
                        Some((first.coeff * constant, first.coeff * quadratic));
                    for term in remaining {
                        let (constant, quadratic) = fused_fold_witness_and_accumulate(
                            &mut term.witness,
                            &self.factor,
                            r_round,
                        );
                        term.cached_accumulate =
                            Some((term.coeff * constant, term.coeff * quadratic));
                    }
                }
                _ => {
                    // With no members, plain folding still binds the shared factor correctly.
                    fold_evals_in_place(&mut self.factor, r_round);
                    for term in &mut self.terms {
                        fold_evals_in_place(&mut term.witness, r_round);
                        term.cached_accumulate = None;
                    }
                }
            }
            return;
        }

        if let Some(&point) = self.extra_point.get(self.extra_round) {
            self.extra_factor_eval *= (E::one() - point) * (E::one() - r_round) + point * r_round;
            self.extra_round += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{One, Prime128Offset275 as E, Ring};

    #[test]
    fn group_rejects_empty_members() {
        let result = ExtensionOpeningReductionGroup::<E>::new(Vec::new(), vec![E::one(); 4]);
        assert!(matches!(result, Err(AkitaError::InvalidInput(message))
            if message == "extension-opening reduction group requires at least one term"));
    }

    #[test]
    fn empty_members_use_plain_fold_for_short_and_long_tables() {
        for len in [2, 4, 8] {
            let factor = (1..=len).map(|value| E::from_u64(value as u64)).collect();
            let term = ExtensionOpeningReductionTerm::new(vec![E::one(); len], E::one());
            let mut group = ExtensionOpeningReductionGroup::new(vec![term], factor).unwrap();
            group.terms.clear();
            group.ingest_challenge(E::from_u64(3));
            let expected = (0..len / 2)
                .map(|pair| E::from_u64((2 * pair + 4) as u64))
                .collect::<Vec<_>>();
            assert_eq!(group.factor, expected);
            assert_eq!(group.num_terms(), 0);
        }
    }

    #[test]
    fn member_order_is_preserved_through_folding_and_cloning() {
        let members = (1..=3)
            .map(|value| ExtensionOpeningReductionTerm::new(vec![E::from_u64(value); 4], E::one()))
            .collect();
        let mut group = ExtensionOpeningReductionGroup::new(members, vec![E::one(); 4]).unwrap();
        assert_eq!(group.num_terms(), 3);
        group.ingest_challenge(E::from_u64(7));
        group.ingest_challenge(E::from_u64(11));
        let expected = (1..=3)
            .map(|value| (E::one(), E::from_u64(value), E::one()))
            .collect::<Vec<_>>();
        assert_eq!(group.final_terms().unwrap(), expected);
        assert_eq!(group.clone().final_terms().unwrap(), expected);
    }
}
