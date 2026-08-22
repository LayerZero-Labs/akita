use akita_serialization::SerializationError;
use core::fmt;
use core::str::FromStr;

/// Exact native profile case carried by a recursion artifact.
///
/// Each value names one verifier monomorphization. The textual form matches
/// the case syntax in `.github/workflows/profile-bench.yml`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AkitaJoltCase {
    /// Scalar fp32 OneHot at nv30 with direct setup contribution.
    OneHotFp32 = 1,
    /// Scalar fp64 OneHot at nv30 with direct setup contribution.
    OneHotFp64 = 2,
    /// Scalar fp128 OneHot at nv36 with direct setup contribution.
    OneHotFp128Direct = 3,
    /// Scalar fp128 OneHot at nv36 with recursive setup contribution.
    OneHotFp128Recursive = 4,
    /// Existing grouped fp128 OneHot recursion benchmark at nv32.
    OneHotFp128MultiGroupRecursive = 5,
}

impl AkitaJoltCase {
    pub(crate) const fn tag(self) -> u8 {
        self as u8
    }

    pub(crate) fn from_tag(tag: u8) -> Result<Self, SerializationError> {
        match tag {
            1 => Ok(Self::OneHotFp32),
            2 => Ok(Self::OneHotFp64),
            3 => Ok(Self::OneHotFp128Direct),
            4 => Ok(Self::OneHotFp128Recursive),
            5 => Ok(Self::OneHotFp128MultiGroupRecursive),
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
            Self::OneHotFp128Direct => "onehot_fp128:36:1:direct",
            Self::OneHotFp128Recursive => "onehot_fp128:36:1:recursive",
            Self::OneHotFp128MultiGroupRecursive => {
                "onehot_fp128_multi_group_recursive:32:4:recursive"
            }
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
            "onehot_fp32:30:1" | "onehot_fp32:30:1:direct" => Ok(Self::OneHotFp32),
            "onehot_fp64:30:1" | "onehot_fp64:30:1:direct" => Ok(Self::OneHotFp64),
            "onehot_fp128:36:1:direct" => Ok(Self::OneHotFp128Direct),
            "onehot_fp128:36:1:recursive" => Ok(Self::OneHotFp128Recursive),
            "onehot_fp128_multi_group_recursive:32:4:recursive" => {
                Ok(Self::OneHotFp128MultiGroupRecursive)
            }
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
            AkitaJoltCase::OneHotFp128Direct,
            AkitaJoltCase::OneHotFp128Recursive,
            AkitaJoltCase::OneHotFp128MultiGroupRecursive,
        ] {
            assert_eq!(case.as_str().parse(), Ok(case));
            assert_eq!(AkitaJoltCase::from_tag(case.tag()).unwrap(), case);
        }
    }

    #[test]
    fn direct_suffix_is_optional_only_for_legacy_three_part_cases() {
        assert_eq!(
            "onehot_fp32:30:1:direct".parse(),
            Ok(AkitaJoltCase::OneHotFp32)
        );
        assert!("onehot_fp128:36:1".parse::<AkitaJoltCase>().is_err());
    }
}
