use super::*;

impl<E: FieldCore> SetupContributionPlan<E> {
    /// Dense setup-weight vector `bar_omega[i]`.
    ///
    /// For each packed setup position `i`, `bar_omega[i]` is the
    /// scalar coefficient multiplying the expanded setup row/ring at that
    /// position in the setup contribution. Each commitment group scatters its
    /// D/B/A segment weights into the shared footprint; overlapping positions,
    /// such as the shared D rows, accumulate additively. For a single group this
    /// equals the historical flat setup-contribution weight vector.
    pub fn materialize_bar_omega(&self) -> Result<Vec<E>, AkitaError> {
        let required = self.required()?;
        let mut bar_omega = vec![E::zero(); required];
        for group in &self.groups {
            let (_, segments) = group.packed_segments(self.d_rows, self.d_physical_cols)?;
            let segment_values = cfg_into_iter!(0..segments.len())
                .map(|idx| {
                    let segment = &segments[idx];
                    let values = (segment.lo..segment.hi)
                        .map(|setup_idx| {
                            segment.weight_at(
                                setup_idx,
                                &group.e_eq_slice,
                                &group.t_eq_slice,
                                &group.z_eq_slice,
                            )
                        })
                        .collect::<Vec<_>>();
                    (segment.lo, values)
                })
                .collect::<Vec<_>>();
            for (lo, values) in segment_values {
                for (offset, value) in values.into_iter().enumerate() {
                    bar_omega[lo + offset] += value;
                }
            }
        }
        Ok(bar_omega)
    }

    /// Eq-weighted setup-weight evaluation
    /// `sum_i eq_setup_idx[i] * bar_omega[i]`.
    ///
    /// This avoids materializing `bar_omega`, but still scans the packed setup
    /// positions. `eq_setup_idx` must have length
    /// `required().next_power_of_two()`.
    pub fn evaluate_bar_omega_with_eq(&self, eq_setup_idx: &[E]) -> Result<E, AkitaError> {
        let required = self.required()?;
        let setup_idx_len = required
            .checked_next_power_of_two()
            .ok_or_else(|| AkitaError::InvalidSetup("setup omega index length overflow".into()))?;
        if eq_setup_idx.len() != setup_idx_len {
            return Err(AkitaError::InvalidSize {
                expected: setup_idx_len,
                actual: eq_setup_idx.len(),
            });
        }
        let mut acc = E::zero();
        for group in &self.groups {
            let (_, segments) = group.packed_segments(self.d_rows, self.d_physical_cols)?;
            let segment_sums: Vec<E> = cfg_into_iter!(0..segments.len())
                .map(|idx| {
                    let segment = &segments[idx];
                    dispatch_segment_roles!(segment, E::zero(), |HAS_D, HAS_B, HAS_A| {
                        group_bar_omega_segment_eval::<E, HAS_D, HAS_B, HAS_A>(
                            segment.lo..segment.hi,
                            eq_setup_idx,
                            segment,
                            &group.e_eq_slice,
                            &group.t_eq_slice,
                            &group.z_eq_slice,
                        )
                    })
                })
                .collect();
            acc += segment_sums.into_iter().sum::<E>();
        }
        Ok(acc)
    }
}
