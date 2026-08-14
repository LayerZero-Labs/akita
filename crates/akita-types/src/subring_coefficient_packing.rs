//! Checked subring coefficient packing geometry.

use akita_challenges::SparseChallengeConfig;
use akita_field::AkitaError;

/// Canonical geometry for one subring coefficient packing opening.
///
/// The A ring dimension satisfies `d_a = k * h * s`. The subring challenge
/// embeds through `Y -> X^(k * h)`, while one partial opening contains `k * s`
/// base field coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubringCoefficientPackingGeometry {
    extension_degree: usize,
    a_ring_dimension: usize,
    challenge_subring_dimension: usize,
    packing_factor: usize,
    subring_embedding_stride: usize,
    partial_base_field_width: usize,
    fold_challenge_config: SparseChallengeConfig,
}

impl SubringCoefficientPackingGeometry {
    /// Construct checked production subring packing geometry.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] unless all dimensions are powers of
    /// two, `k * s` divides `d_a`, and `s` has a production sparse challenge
    /// family that passes the existing entropy audit.
    pub fn try_new(
        extension_degree: usize,
        a_ring_dimension: usize,
        challenge_subring_dimension: usize,
    ) -> Result<Self, AkitaError> {
        if !extension_degree.is_power_of_two()
            || !a_ring_dimension.is_power_of_two()
            || !challenge_subring_dimension.is_power_of_two()
        {
            return Err(AkitaError::InvalidSetup(
                "subring packing dimensions must be nonzero powers of two".into(),
            ));
        }

        let fold_challenge_config = SparseChallengeConfig::production_for_ring_dim(
            challenge_subring_dimension,
        )
        .ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "challenge subring dimension {challenge_subring_dimension} has no production challenge family"
            ))
        })?;
        fold_challenge_config
            .validate_for_ring_dim(challenge_subring_dimension)
            .map_err(|reason| {
                AkitaError::InvalidSetup(format!(
                    "subring challenge family fails its entropy audit: {reason}"
                ))
            })?;

        let partial_base_field_width = extension_degree
            .checked_mul(challenge_subring_dimension)
            .ok_or_else(|| AkitaError::InvalidSetup("partial opening width overflow".into()))?;
        if !a_ring_dimension.is_multiple_of(partial_base_field_width) {
            return Err(AkitaError::InvalidSetup(format!(
                "partial opening width {partial_base_field_width} does not divide A ring dimension {a_ring_dimension}"
            )));
        }
        let packing_factor = a_ring_dimension / partial_base_field_width;
        let subring_embedding_stride = extension_degree
            .checked_mul(packing_factor)
            .ok_or_else(|| AkitaError::InvalidSetup("subring embedding stride overflow".into()))?;

        Ok(Self {
            extension_degree,
            a_ring_dimension,
            challenge_subring_dimension,
            packing_factor,
            subring_embedding_stride,
            partial_base_field_width,
            fold_challenge_config,
        })
    }

    /// Extension degree `k = [E:K]`.
    #[must_use]
    pub const fn extension_degree(self) -> usize {
        self.extension_degree
    }

    /// A ring dimension `d_a`.
    #[must_use]
    pub const fn a_ring_dimension(self) -> usize {
        self.a_ring_dimension
    }

    /// Challenge subring dimension `s`.
    #[must_use]
    pub const fn challenge_subring_dimension(self) -> usize {
        self.challenge_subring_dimension
    }

    /// Packing factor `h = d_a / (k * s)`.
    #[must_use]
    pub const fn packing_factor(self) -> usize {
        self.packing_factor
    }

    /// Exponent stride `k * h` in the embedding `Y -> X^(k * h)`.
    #[must_use]
    pub const fn subring_embedding_stride(self) -> usize {
        self.subring_embedding_stride
    }

    /// Base field coordinate width `k * s` of one partial opening.
    #[must_use]
    pub const fn partial_base_field_width(self) -> usize {
        self.partial_base_field_width
    }

    /// Production sparse challenge family fixed by `s`.
    #[must_use]
    pub const fn fold_challenge_config(self) -> SparseChallengeConfig {
        self.fold_challenge_config
    }

    /// Flatten `(low_coefficient_index, subring_coefficient_index)` into one A
    /// ring coefficient index.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when either coordinate is outside
    /// the checked geometry.
    pub fn a_ring_coefficient_index(
        self,
        low_coefficient_index: usize,
        subring_coefficient_index: usize,
    ) -> Result<usize, AkitaError> {
        if low_coefficient_index >= self.subring_embedding_stride
            || subring_coefficient_index >= self.challenge_subring_dimension
        {
            return Err(AkitaError::InvalidSetup(
                "subring coefficient coordinates are outside the A ring".into(),
            ));
        }
        subring_coefficient_index
            .checked_mul(self.subring_embedding_stride)
            .and_then(|offset| offset.checked_add(low_coefficient_index))
            .filter(|&index| index < self.a_ring_dimension)
            .ok_or_else(|| AkitaError::InvalidSetup("A ring coefficient index overflow".into()))
    }

    /// Split one A ring coefficient into
    /// `(low_coefficient_index, subring_coefficient_index)`.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when `index` is outside the A ring.
    pub fn a_ring_coefficient_coordinates(
        self,
        index: usize,
    ) -> Result<(usize, usize), AkitaError> {
        if index >= self.a_ring_dimension {
            return Err(AkitaError::InvalidSetup(
                "A ring coefficient lies outside subring packing geometry".into(),
            ));
        }
        Ok((
            index % self.subring_embedding_stride,
            index / self.subring_embedding_stride,
        ))
    }

    /// Flatten `(extension_coordinate, subring_coefficient_index)` in the
    /// canonical partial opening layout.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when either coordinate is outside
    /// the checked geometry.
    pub fn partial_base_field_coordinate_index(
        self,
        extension_coordinate: usize,
        subring_coefficient_index: usize,
    ) -> Result<usize, AkitaError> {
        if extension_coordinate >= self.extension_degree
            || subring_coefficient_index >= self.challenge_subring_dimension
        {
            return Err(AkitaError::InvalidSetup(
                "partial opening coordinates are outside subring packing geometry".into(),
            ));
        }
        extension_coordinate
            .checked_mul(self.challenge_subring_dimension)
            .and_then(|offset| offset.checked_add(subring_coefficient_index))
            .filter(|&index| index < self.partial_base_field_width)
            .ok_or_else(|| AkitaError::InvalidSetup("partial opening index overflow".into()))
    }

    /// Split one partial opening index into
    /// `(extension_coordinate, subring_coefficient_index)`.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when `index` is outside the partial
    /// opening.
    pub fn partial_base_field_coordinates(
        self,
        index: usize,
    ) -> Result<(usize, usize), AkitaError> {
        if index >= self.partial_base_field_width {
            return Err(AkitaError::InvalidSetup(
                "partial opening coefficient lies outside subring packing geometry".into(),
            ));
        }
        Ok((
            index / self.challenge_subring_dimension,
            index % self.challenge_subring_dimension,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::PRODUCTION_FOLD_CHALLENGE_RING_DIMS;

    #[test]
    fn derives_every_production_subring_packing_geometry() {
        for &s in PRODUCTION_FOLD_CHALLENGE_RING_DIMS {
            for k in [1usize, 2, 4] {
                for h in [1usize, 2, 4] {
                    let d_a = k * h * s;
                    let geometry =
                        SubringCoefficientPackingGeometry::try_new(k, d_a, s).expect("geometry");
                    assert_eq!(geometry.extension_degree(), k);
                    assert_eq!(geometry.a_ring_dimension(), d_a);
                    assert_eq!(geometry.challenge_subring_dimension(), s);
                    assert_eq!(geometry.packing_factor(), h);
                    assert_eq!(geometry.subring_embedding_stride(), k * h);
                    assert_eq!(geometry.partial_base_field_width(), k * s);
                    assert!(geometry
                        .fold_challenge_config()
                        .matches_production_ladder(s));
                }
            }
        }
    }

    #[test]
    fn rejects_malformed_or_unregistered_geometry() {
        for (k, d_a, s) in [
            (0, 256, 64),
            (3, 384, 64),
            (2, 0, 64),
            (2, 192, 64),
            (2, 256, 0),
            (2, 256, 32),
            (4, 128, 64),
        ] {
            assert!(
                SubringCoefficientPackingGeometry::try_new(k, d_a, s).is_err(),
                "{k}/{d_a}/{s}"
            );
        }

        let high_bit = 1usize << (usize::BITS - 1);
        assert!(SubringCoefficientPackingGeometry::try_new(high_bit, high_bit, 64).is_err());
    }

    #[test]
    fn a_ring_coefficient_indices_round_trip() {
        for &s in PRODUCTION_FOLD_CHALLENGE_RING_DIMS {
            for k in [1usize, 2, 4] {
                for h in [1usize, 2, 4] {
                    let geometry = SubringCoefficientPackingGeometry::try_new(k, k * h * s, s)
                        .expect("geometry");
                    for index in 0..geometry.a_ring_dimension() {
                        let (low_coefficient_index, subring_coefficient_index) = geometry
                            .a_ring_coefficient_coordinates(index)
                            .expect("coordinates");
                        assert_eq!(
                            geometry
                                .a_ring_coefficient_index(
                                    low_coefficient_index,
                                    subring_coefficient_index,
                                )
                                .expect("index"),
                            index
                        );
                        assert_eq!(
                            index,
                            low_coefficient_index
                                + geometry.subring_embedding_stride() * subring_coefficient_index
                        );
                    }
                    assert!(geometry
                        .a_ring_coefficient_index(geometry.subring_embedding_stride(), 0)
                        .is_err());
                    assert!(geometry
                        .a_ring_coefficient_index(0, geometry.challenge_subring_dimension())
                        .is_err());
                    assert!(geometry
                        .a_ring_coefficient_coordinates(geometry.a_ring_dimension())
                        .is_err());
                }
            }
        }
    }

    #[test]
    fn partial_base_field_coordinates_round_trip() {
        for &s in PRODUCTION_FOLD_CHALLENGE_RING_DIMS {
            for k in [1usize, 2, 4] {
                for h in [1usize, 2, 4] {
                    let geometry = SubringCoefficientPackingGeometry::try_new(k, k * h * s, s)
                        .expect("geometry");
                    for index in 0..geometry.partial_base_field_width() {
                        let (extension_coordinate, subring_coefficient_index) = geometry
                            .partial_base_field_coordinates(index)
                            .expect("coordinates");
                        assert_eq!(
                            geometry
                                .partial_base_field_coordinate_index(
                                    extension_coordinate,
                                    subring_coefficient_index,
                                )
                                .expect("index"),
                            index
                        );
                        assert_eq!(index, extension_coordinate * s + subring_coefficient_index);
                    }
                    assert!(geometry
                        .partial_base_field_coordinate_index(geometry.extension_degree(), 0)
                        .is_err());
                    assert!(geometry
                        .partial_base_field_coordinate_index(
                            0,
                            geometry.challenge_subring_dimension()
                        )
                        .is_err());
                    assert!(geometry
                        .partial_base_field_coordinates(geometry.partial_base_field_width())
                        .is_err());
                }
            }
        }
    }
}
