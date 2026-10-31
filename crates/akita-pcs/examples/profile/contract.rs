//! Canonical profile workload shapes, checked before allocating witnesses.
use akita_config::CommitmentConfig;
use serde::Deserialize;
use std::num::NonZeroUsize;

/// A required JSON field whose explicit null value has a defined meaning.
/// Unlike `Option`, a missing declaration field is a deserialization error.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Nullable<T> {
    Null(()),
    Value(T),
}

impl Nullable<Vec<NonZeroUsize>> {
    fn admits(&self, value: usize) -> bool {
        value > 0
            && match self {
                Self::Null(()) => true,
                Self::Value(values) => values.iter().any(|allowed| allowed.get() == value),
            }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Declaration {
    mode: String,
    setup_mode: String,
    family: String,
    num_vars: Nullable<Vec<NonZeroUsize>>,
    num_polys: Nullable<Vec<NonZeroUsize>>,
    precommitted: Vec<[NonZeroUsize; 2]>,
    final_polys: Nullable<NonZeroUsize>,
}

/// The executable topology after declaration, shape, and config validation.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Workload {
    SingleGroup,
    MultiGroup {
        pre_num_vars: usize,
        final_num_polys: usize,
    },
}

impl Declaration {
    fn validate<Cfg: CommitmentConfig>(
        &self,
        nv: usize,
        np: usize,
    ) -> Result<Workload, &'static str> {
        if !self.num_vars.admits(nv) || !self.num_polys.admits(np) {
            return Err("unsupported profile workload shape");
        }
        if self.family != Cfg::schedule_family_name() {
            return Err("profile dispatch/catalog mismatch");
        }
        match (self.precommitted.as_slice(), &self.final_polys) {
            ([], Nullable::Null(())) => Ok(Workload::SingleGroup),
            ([first, second], Nullable::Value(final_np))
                if first == second
                    && first[1].get() == 1
                    && akita_error::checked::sum([2, final_np.get()]) == Some(np) =>
            {
                Ok(Workload::MultiGroup {
                    pre_num_vars: first[0].get(),
                    final_num_polys: final_np.get(),
                })
            }
            _ => Err("unsupported profile group topology"),
        }
    }
}

pub(crate) fn attest<Cfg: CommitmentConfig>(mode: &str, nv: usize, np: usize) -> Workload {
    let setup = std::env::var("AKITA_SETUP_MODE").unwrap_or_else(|_| "direct".into());
    let rows: Vec<Declaration> = serde_json::from_str(include_str!("workloads.json"))
        .expect("profile workload declarations");
    let row = rows
        .into_iter()
        .find(|row| row.mode == mode && row.setup_mode == setup)
        .expect("unsupported profile mode/setup combination");
    let workload = row
        .validate::<Cfg>(nv, np)
        .expect("profile workload contract");
    let source_k = akita_config::unit_onehot_source_chunk_size::<Cfg>()
        .ok()
        .map(|k| k.min(akita_error::checked::pow2(nv).expect("profile arity")));
    let final_polys = match workload {
        Workload::SingleGroup => None,
        Workload::MultiGroup {
            final_num_polys, ..
        } => Some(final_num_polys),
    };
    // JSON is only the metadata output format; execution consumes Workload.
    let metadata = serde_json::json!({"mode": mode, "setup_mode": setup,
        "family": Cfg::schedule_family_name(), "num_vars": nv, "num_polys": np,
        "onehot_k": source_k, "precommitted": row.precommitted,
        "final_polys": final_polys});
    if std::env::var("AKITA_PROFILE_CONTRACT_ONLY").as_deref() == Ok("1") {
        println!("{metadata}");
        std::process::exit(0);
    }
    tracing::info!(metadata = %metadata, "profile workload contract");
    workload
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_config::proof_optimized::{fp128, fp32};

    fn dense() -> Declaration {
        serde_json::from_str(
            r#"{"mode":"dense_fp32","setup_mode":"direct",
            "family":"fp32_dense","num_vars":[30],"num_polys":[1],
            "precommitted":[],"final_polys":null}"#,
        )
        .unwrap()
    }

    fn grouped() -> Declaration {
        serde_json::from_str(
            r#"{"mode":"onehot_fp128_multi_group","setup_mode":"direct",
            "family":"fp128_onehot","num_vars":[31],"num_polys":[4],
            "precommitted":[[16,1],[16,1]],"final_polys":2}"#,
        )
        .unwrap()
    }

    #[test]
    fn wrong_family_is_rejected_even_when_both_catalogs_have_the_shape() {
        let mut row = dense();
        row.family = "fp32_onehot".into();
        assert_eq!(
            row.validate::<fp32::Dense>(30, 1),
            Err("profile dispatch/catalog mismatch")
        );
    }

    #[test]
    fn shape_is_checked_before_any_witness_allocation() {
        let row = dense();
        assert!(row.validate::<fp32::Dense>(31, 1).is_err());
        assert_eq!(
            row.validate::<fp32::Dense>(30, 1),
            Ok(Workload::SingleGroup)
        );
    }

    #[test]
    fn missing_shape_fields_are_not_unrestricted() {
        let complete = serde_json::json!({"mode":"dense_fp32",
            "setup_mode":"direct","family":"fp32_dense","num_vars":null,
            "num_polys":null,"precommitted":[],"final_polys":null});
        let explicit: Declaration = serde_json::from_value(complete.clone()).unwrap();
        assert_eq!(
            explicit.validate::<fp32::Dense>(30, 1),
            Ok(Workload::SingleGroup)
        );
        for key in ["num_vars", "num_polys", "precommitted", "final_polys"] {
            let mut missing = complete.clone();
            missing.as_object_mut().unwrap().remove(key);
            assert!(
                serde_json::from_value::<Declaration>(missing).is_err(),
                "{key}"
            );
        }
    }

    #[test]
    fn grouped_driver_receives_validated_numbers() {
        assert_eq!(
            grouped().validate::<fp128::OneHot>(31, 4),
            Ok(Workload::MultiGroup {
                pre_num_vars: 16,
                final_num_polys: 2
            })
        );
    }

    #[test]
    fn inconsistent_grouped_topologies_are_rejected() {
        let mut row = grouped();
        row.precommitted[1][0] = NonZeroUsize::new(17).unwrap();
        assert!(row.validate::<fp128::OneHot>(31, 4).is_err());
        let mut row = grouped();
        row.precommitted[0][1] = NonZeroUsize::new(2).unwrap();
        row.precommitted[1] = row.precommitted[0];
        assert!(row.validate::<fp128::OneHot>(31, 4).is_err());
        let mut row = grouped();
        row.final_polys = Nullable::Value(NonZeroUsize::new(3).unwrap());
        assert!(row.validate::<fp128::OneHot>(31, 4).is_err());
        let mut row = grouped();
        row.precommitted.pop();
        assert!(row.validate::<fp128::OneHot>(31, 4).is_err());
    }

    #[test]
    fn declarations_reject_non_integral_and_zero_sizes() {
        for value in [
            serde_json::json!([0]),
            serde_json::json!([1.5]),
            serde_json::json!("all"),
        ] {
            let mut row = serde_json::json!({"mode":"dense_fp32","setup_mode":"direct",
                "family":"fp32_dense","num_vars":[30],"num_polys":[1],
                "precommitted":[],"final_polys":null});
            row["num_vars"] = value;
            assert!(serde_json::from_value::<Declaration>(row).is_err());
        }
    }
}
