use crate::CanonicalField;

pub(super) fn response_within_bound<F: CanonicalField>(
    bound: Option<u128>,
    response: &[F],
) -> bool {
    let Some(bound) = bound else {
        return true;
    };
    let modulus = field_modulus::<F>();
    response
        .iter()
        .all(|&value| centered_abs(value, modulus) <= bound)
}

fn field_modulus<F: CanonicalField>() -> u128 {
    (-F::one()).to_canonical_u128() + 1
}

fn centered_abs<F: CanonicalField>(value: F, modulus: u128) -> u128 {
    let canonical = value.to_canonical_u128();
    if canonical == 0 {
        0
    } else {
        canonical.min(modulus - canonical)
    }
}
