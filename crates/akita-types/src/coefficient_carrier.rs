//! Checked coefficient carrier geometry.

use akita_challenges::SparseChallengeConfig;
use akita_field::AkitaError;

/// Canonical geometry for one coefficient carrier opening.
///
/// The ambient ring dimension satisfies `d_a = k * h * s`. The carrier
/// challenge embeds through `Y -> X^(k * h)`, while one direct partial contains
/// `k * s` base field coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CarrierGeometry {
    extension_degree: usize,
    ambient_ring_dimension: usize,
    carrier_ring_dimension: usize,
    packing_gain: usize,
    ambient_stride: usize,
    partial_coordinate_width: usize,
    challenge_config: SparseChallengeConfig,
}

impl CarrierGeometry {
    /// Construct checked production carrier geometry.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] unless all dimensions are powers of
    /// two, `k * s` divides `d_a`, and `s` has a production sparse challenge
    /// family that passes the existing entropy audit.
    pub fn try_new(
        extension_degree: usize,
        ambient_ring_dimension: usize,
        carrier_ring_dimension: usize,
    ) -> Result<Self, AkitaError> {
        if !extension_degree.is_power_of_two()
            || !ambient_ring_dimension.is_power_of_two()
            || !carrier_ring_dimension.is_power_of_two()
        {
            return Err(AkitaError::InvalidSetup(
                "carrier dimensions must be nonzero powers of two".into(),
            ));
        }

        let challenge_config = SparseChallengeConfig::production_for_ring_dim(
            carrier_ring_dimension,
        )
        .ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "carrier dimension {carrier_ring_dimension} has no production challenge family"
            ))
        })?;
        challenge_config
            .validate_for_ring_dim(carrier_ring_dimension)
            .map_err(|reason| {
                AkitaError::InvalidSetup(format!(
                    "carrier challenge family fails its entropy audit: {reason}"
                ))
            })?;

        let partial_coordinate_width = extension_degree
            .checked_mul(carrier_ring_dimension)
            .ok_or_else(|| AkitaError::InvalidSetup("carrier partial width overflow".into()))?;
        if !ambient_ring_dimension.is_multiple_of(partial_coordinate_width) {
            return Err(AkitaError::InvalidSetup(format!(
                "carrier partial width {partial_coordinate_width} does not divide ambient ring dimension {ambient_ring_dimension}"
            )));
        }
        let packing_gain = ambient_ring_dimension / partial_coordinate_width;
        let ambient_stride = extension_degree
            .checked_mul(packing_gain)
            .ok_or_else(|| AkitaError::InvalidSetup("carrier embedding stride overflow".into()))?;

        Ok(Self {
            extension_degree,
            ambient_ring_dimension,
            carrier_ring_dimension,
            packing_gain,
            ambient_stride,
            partial_coordinate_width,
            challenge_config,
        })
    }

    /// Extension degree `k = [E:K]`.
    #[must_use]
    pub const fn extension_degree(self) -> usize {
        self.extension_degree
    }

    /// Ambient A ring dimension `d_a`.
    #[must_use]
    pub const fn ambient_ring_dimension(self) -> usize {
        self.ambient_ring_dimension
    }

    /// Carrier ring dimension `s`.
    #[must_use]
    pub const fn carrier_ring_dimension(self) -> usize {
        self.carrier_ring_dimension
    }

    /// Packing gain `h = d_a / (k * s)`.
    #[must_use]
    pub const fn packing_gain(self) -> usize {
        self.packing_gain
    }

    /// Exponent stride `k * h` in the embedding `Y -> X^(k * h)`.
    #[must_use]
    pub const fn ambient_stride(self) -> usize {
        self.ambient_stride
    }

    /// Base field coordinate width `k * s` of one direct partial.
    #[must_use]
    pub const fn partial_coordinate_width(self) -> usize {
        self.partial_coordinate_width
    }

    /// Production sparse challenge family fixed by `s`.
    #[must_use]
    pub const fn challenge_config(self) -> SparseChallengeConfig {
        self.challenge_config
    }

    /// Flatten `(ambient_lane, carrier_coefficient)` into one ambient ring
    /// coefficient index.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when either coordinate is outside
    /// the checked geometry.
    pub fn ambient_coefficient_index(
        self,
        ambient_lane: usize,
        carrier_coefficient: usize,
    ) -> Result<usize, AkitaError> {
        if ambient_lane >= self.ambient_stride || carrier_coefficient >= self.carrier_ring_dimension
        {
            return Err(AkitaError::InvalidSetup(
                "carrier coefficient coordinates are outside the ambient ring".into(),
            ));
        }
        carrier_coefficient
            .checked_mul(self.ambient_stride)
            .and_then(|offset| offset.checked_add(ambient_lane))
            .filter(|&index| index < self.ambient_ring_dimension)
            .ok_or_else(|| AkitaError::InvalidSetup("ambient carrier index overflow".into()))
    }

    /// Split one ambient ring coefficient into
    /// `(ambient_lane, carrier_coefficient)`.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when `index` is outside the ambient
    /// ring.
    pub fn ambient_coefficient_coordinates(
        self,
        index: usize,
    ) -> Result<(usize, usize), AkitaError> {
        if index >= self.ambient_ring_dimension {
            return Err(AkitaError::InvalidSetup(
                "ambient coefficient lies outside carrier geometry".into(),
            ));
        }
        Ok((index % self.ambient_stride, index / self.ambient_stride))
    }

    /// Flatten `(extension_coordinate, carrier_coefficient)` in the canonical
    /// direct partial layout.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when either coordinate is outside
    /// the checked geometry.
    pub fn partial_coordinate_index(
        self,
        extension_coordinate: usize,
        carrier_coefficient: usize,
    ) -> Result<usize, AkitaError> {
        if extension_coordinate >= self.extension_degree
            || carrier_coefficient >= self.carrier_ring_dimension
        {
            return Err(AkitaError::InvalidSetup(
                "direct partial coordinates are outside carrier geometry".into(),
            ));
        }
        extension_coordinate
            .checked_mul(self.carrier_ring_dimension)
            .and_then(|offset| offset.checked_add(carrier_coefficient))
            .filter(|&index| index < self.partial_coordinate_width)
            .ok_or_else(|| AkitaError::InvalidSetup("direct partial index overflow".into()))
    }

    /// Split one direct partial index into
    /// `(extension_coordinate, carrier_coefficient)`.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when `index` is outside the direct
    /// partial.
    pub fn partial_coordinate_coordinates(
        self,
        index: usize,
    ) -> Result<(usize, usize), AkitaError> {
        if index >= self.partial_coordinate_width {
            return Err(AkitaError::InvalidSetup(
                "direct partial coefficient lies outside carrier geometry".into(),
            ));
        }
        Ok((
            index / self.carrier_ring_dimension,
            index % self.carrier_ring_dimension,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::PRODUCTION_FOLD_CHALLENGE_RING_DIMS;

    #[test]
    fn derives_every_production_carrier_geometry() {
        for &s in PRODUCTION_FOLD_CHALLENGE_RING_DIMS {
            for k in [1usize, 2, 4] {
                for h in [1usize, 2, 4] {
                    let d_a = k * h * s;
                    let geometry = CarrierGeometry::try_new(k, d_a, s).expect("geometry");
                    assert_eq!(geometry.extension_degree(), k);
                    assert_eq!(geometry.ambient_ring_dimension(), d_a);
                    assert_eq!(geometry.carrier_ring_dimension(), s);
                    assert_eq!(geometry.packing_gain(), h);
                    assert_eq!(geometry.ambient_stride(), k * h);
                    assert_eq!(geometry.partial_coordinate_width(), k * s);
                    assert!(geometry.challenge_config().matches_production_ladder(s));
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
                CarrierGeometry::try_new(k, d_a, s).is_err(),
                "{k}/{d_a}/{s}"
            );
        }

        let high_bit = 1usize << (usize::BITS - 1);
        assert!(CarrierGeometry::try_new(high_bit, high_bit, 64).is_err());
    }

    #[test]
    fn ambient_coefficient_indices_round_trip() {
        for &s in PRODUCTION_FOLD_CHALLENGE_RING_DIMS {
            for k in [1usize, 2, 4] {
                for h in [1usize, 2, 4] {
                    let geometry = CarrierGeometry::try_new(k, k * h * s, s).expect("geometry");
                    for index in 0..geometry.ambient_ring_dimension() {
                        let (lane, carrier) = geometry
                            .ambient_coefficient_coordinates(index)
                            .expect("coordinates");
                        assert_eq!(
                            geometry
                                .ambient_coefficient_index(lane, carrier)
                                .expect("index"),
                            index
                        );
                        assert_eq!(index, lane + geometry.ambient_stride() * carrier);
                    }
                    assert!(geometry
                        .ambient_coefficient_index(geometry.ambient_stride(), 0)
                        .is_err());
                    assert!(geometry
                        .ambient_coefficient_index(0, geometry.carrier_ring_dimension())
                        .is_err());
                    assert!(geometry
                        .ambient_coefficient_coordinates(geometry.ambient_ring_dimension())
                        .is_err());
                }
            }
        }
    }

    #[test]
    fn direct_partial_indices_round_trip() {
        for &s in PRODUCTION_FOLD_CHALLENGE_RING_DIMS {
            for k in [1usize, 2, 4] {
                for h in [1usize, 2, 4] {
                    let geometry = CarrierGeometry::try_new(k, k * h * s, s).expect("geometry");
                    for index in 0..geometry.partial_coordinate_width() {
                        let (extension, carrier) = geometry
                            .partial_coordinate_coordinates(index)
                            .expect("coordinates");
                        assert_eq!(
                            geometry
                                .partial_coordinate_index(extension, carrier)
                                .expect("index"),
                            index
                        );
                        assert_eq!(index, extension * s + carrier);
                    }
                    assert!(geometry
                        .partial_coordinate_index(geometry.extension_degree(), 0)
                        .is_err());
                    assert!(geometry
                        .partial_coordinate_index(0, geometry.carrier_ring_dimension())
                        .is_err());
                    assert!(geometry
                        .partial_coordinate_coordinates(geometry.partial_coordinate_width())
                        .is_err());
                }
            }
        }
    }
}
