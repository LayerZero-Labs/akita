//! Typed source-batched S11a readings admitted by the research driver.

use akita_error::AkitaError;
use akita_params::binary_setup::BinaryFieldProfile;
use jolt_field::{CanonicalEncoding, Field};

/// A point and its ordered owner evaluations. The field is a type parameter,
/// not a residue cast; the extension-plane adapter can retain E64 coordinates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResearchTerminal<E> {
    pub point: Vec<E>,
    pub evaluations: Vec<E>,
}

/// New route-1 work uses E64. Base mode retains the measured v2 proof control.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResearchTerminalMode {
    BaseS11a,
    E64TwoPointS11a,
}

/// Ordered readings per source. This container retains the field type and point
/// multiplicity; both E64 readings must authenticate every owner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResearchTerminals<E> {
    pub mode: ResearchTerminalMode,
    pub readings: Vec<ResearchTerminal<E>>,
}

impl<E: Field> ResearchTerminals<E> {
    /// Validate the complete inventory without dropping a second reading.
    pub fn validate<F: Field + CanonicalEncoding>(
        &self,
    ) -> Result<&[ResearchTerminal<E>], AkitaError>
    where
        E: jolt_field::ExtField<F>,
    {
        let expected = match self.mode {
            ResearchTerminalMode::BaseS11a => 1,
            ResearchTerminalMode::E64TwoPointS11a => {
                BinaryFieldProfile::P64
                    .check_field_pair(akita_params::field_modulus::<F>()?, E::DEGREE)?;
                2
            }
        };
        if self.readings.len() != expected {
            return Err(AkitaError::InvalidInput("research point count".into()));
        }
        for (index, terminal) in self.readings.iter().enumerate() {
            if self
                .readings
                .iter()
                .take(index)
                .any(|earlier| earlier.point == terminal.point)
            {
                return Err(AkitaError::InvalidInput("duplicate research point".into()));
            }
            if self.mode == ResearchTerminalMode::BaseS11a
                && terminal
                    .point
                    .iter()
                    .chain(&terminal.evaluations)
                    .any(|value| {
                        (1..E::DEGREE)
                            .any(|coordinate| !value.base_coefficient(coordinate).is_zero())
                    })
            {
                return Err(AkitaError::InvalidInput(
                    "research base terminal has E64 data".into(),
                ));
            }
        }
        Ok(&self.readings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Ext2, ExtField, Prime64Offset59 as F, Ring, Zero};

    #[test]
    fn research_terminal_inventory_rejects_unconsumed_points_and_nonbase_coordinates() {
        type E = Ext2<F>;
        let terminal = ResearchTerminal {
            point: vec![E::lift_base(F::from_u64(7))],
            evaluations: vec![E::zero()],
        };
        let mut terminals = ResearchTerminals {
            mode: ResearchTerminalMode::BaseS11a,
            readings: vec![terminal.clone()],
        };
        terminals.validate::<F>().unwrap();
        terminals.readings.push(terminal);
        assert!(terminals.validate::<F>().is_err());
        terminals.readings.pop();
        terminals.readings[0].point[0] =
            E::from_base_fn(|index| F::from_u64(if index == 0 { 7 } else { 1 }));
        assert!(terminals.validate::<F>().is_err());
        terminals.readings.clear();
        assert!(terminals.validate::<F>().is_err());
    }

    #[test]
    fn research_e64_inventory_requires_two_distinct_points_and_the_e64_field() {
        type E = Ext2<F>;
        let terminal = |value| ResearchTerminal {
            point: vec![E::from_base_fn(|coordinate| {
                F::from_u64(value + u64::try_from(coordinate).unwrap())
            })],
            evaluations: vec![E::zero()],
        };
        let mut terminals = ResearchTerminals {
            mode: ResearchTerminalMode::E64TwoPointS11a,
            readings: vec![terminal(7), terminal(11)],
        };
        assert_eq!(terminals.validate::<F>().unwrap().len(), 2);
        terminals.readings[1] = terminals.readings[0].clone();
        assert!(terminals.validate::<F>().is_err());
        terminals.readings.pop();
        assert!(terminals.validate::<F>().is_err());
        terminals.readings.extend([terminal(11), terminal(13)]);
        assert!(terminals.validate::<F>().is_err());
        let wrong_field = ResearchTerminals {
            mode: ResearchTerminalMode::E64TwoPointS11a,
            readings: vec![
                ResearchTerminal {
                    point: vec![F::from_u64(3)],
                    evaluations: vec![F::zero()],
                };
                2
            ],
        };
        assert!(wrong_field.validate::<F>().is_err());
    }
}
