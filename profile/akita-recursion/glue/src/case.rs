use akita_serialization::SerializationError;
use core::fmt;
use core::str::FromStr;

/// Exact native profile case carried by a recursion artifact.
///
/// Each value names one verifier monomorphization. The textual form matches
/// the case syntax in `.github/workflows/profile-bench.yml`.
/// Tag 3 (the retired direct-only fp128 case) stays unused; tag 4 retains
/// the offloading case identity. Older family frames still require admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AkitaJoltCase {
    /// Scalar fp32 OneHot at nv30 with planner-selected setup offloading.
    OneHotFp32 = 1,
    /// Scalar fp64 OneHot at nv30 with planner-selected setup offloading.
    OneHotFp64 = 2,
    /// Scalar fp128 OneHot at nv36 with planner-selected setup offloading.
    OneHotFp128 = 4,
    /// Existing grouped fp128 OneHot recursion benchmark at nv32.
    OneHotFp128MultiGroup = 5,
}

impl AkitaJoltCase {
    pub(crate) const fn tag(self) -> u8 {
        self as u8
    }

    pub(crate) fn from_tag(tag: u8) -> Result<Self, SerializationError> {
        match tag {
            1 => Ok(Self::OneHotFp32),
            2 => Ok(Self::OneHotFp64),
            4 => Ok(Self::OneHotFp128),
            5 => Ok(Self::OneHotFp128MultiGroup),
            _ => Err(SerializationError::InvalidData(format!(
                "akita-jolt blob has unknown CI case tag {tag}"
            ))),
        }
    }

    /// Canonical CI case string.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OneHotFp32 => "onehot_fp32:30:1",
            Self::OneHotFp64 => "onehot_fp64:30:1",
            Self::OneHotFp128 => "onehot_fp128:36:1",
            Self::OneHotFp128MultiGroup => "onehot_fp128_multi_group:32:4",
        }
    }
}

impl fmt::Display for AkitaJoltCase {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for AkitaJoltCase {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "onehot_fp32:30:1" => Ok(Self::OneHotFp32),
            "onehot_fp64:30:1" => Ok(Self::OneHotFp64),
            "onehot_fp128:36:1" => Ok(Self::OneHotFp128),
            "onehot_fp128_multi_group:32:4" => Ok(Self::OneHotFp128MultiGroup),
            _ => Err(format!("unsupported Akita Jolt CI case `{value}`")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_case_strings_round_trip() {
        for case in [
            AkitaJoltCase::OneHotFp32,
            AkitaJoltCase::OneHotFp64,
            AkitaJoltCase::OneHotFp128,
            AkitaJoltCase::OneHotFp128MultiGroup,
        ] {
            assert_eq!(case.as_str().parse(), Ok(case));
            assert_eq!(AkitaJoltCase::from_tag(case.tag()).unwrap(), case);
        }
    }

    #[test]
    fn retired_mode_suffixes_and_direct_case_tag_are_rejected() {
        assert!("onehot_fp32:30:1:direct".parse::<AkitaJoltCase>().is_err());
        assert!("onehot_fp128:36:1:recursive"
            .parse::<AkitaJoltCase>()
            .is_err());
        assert_eq!("onehot_fp128:36:1".parse(), Ok(AkitaJoltCase::OneHotFp128));
        assert!(AkitaJoltCase::from_tag(3).is_err());
    }
}
