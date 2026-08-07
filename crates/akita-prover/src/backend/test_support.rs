use crate::DecomposeFoldWitness;
use akita_field::FieldCore;

pub(crate) fn aggregate_witnesses<F: FieldCore, const D: usize>(
    witnesses: &[DecomposeFoldWitness<F>],
) -> DecomposeFoldWitness<F> {
    let Some((first, rest)) = witnesses.split_first() else {
        panic!("aggregate_witnesses requires at least one witness");
    };
    first
        .ensure_ring_dim::<D>()
        .expect("witness ring dimension");
    let mut z_folded_rings = first.z_folded_rings_trusted::<D>().to_vec();
    let mut centered_coeffs = first.centered_coeffs_owned::<D>();

    for witness in rest {
        witness
            .ensure_ring_dim::<D>()
            .expect("witness ring dimension");
        for (dst, src) in z_folded_rings
            .iter_mut()
            .zip(witness.z_folded_rings_trusted::<D>())
        {
            *dst += *src;
        }
        for (dst, src) in centered_coeffs
            .iter_mut()
            .zip(witness.centered_coeffs_trusted::<D>())
        {
            for k in 0..D {
                dst[k] = dst[k]
                    .checked_add(src[k])
                    .expect("centered coefficient overflow");
            }
        }
    }

    let centered_inf_norm = centered_coeffs
        .iter()
        .flat_map(|coeffs| coeffs.iter())
        .map(|coeff| coeff.unsigned_abs())
        .max()
        .unwrap_or(0);

    DecomposeFoldWitness::from_parts(z_folded_rings, centered_coeffs, centered_inf_norm)
}
