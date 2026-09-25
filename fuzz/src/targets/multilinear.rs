//! Equality tables, basis weights, multilinear evaluation, and folding.
//!
//! Oracle: the per-index product definition of the Lagrange and monomial
//! weights in little-endian variable order (`oracle::index_weight`), which
//! shares no code with `EqPolynomial`'s recurrences, and direct pairwise
//! folds.

use crate::input::Reader;
use crate::oracle::{self, index_weight};
use crate::{gen, stats};
use akita_algebra::{EqPolynomial, GruenSplitEq, SplitEqEvals};
use akita_config::proof_optimized::{fp128, fp32, fp64};
use akita_error::AkitaError;
use akita_types::{
    basis_weights, basis_weights_prefix, lagrange_weights, monomial_weights, BasisMode,
};
use jolt_field::{CanonicalEncoding, ExtField, Field, Fold};
use jolt_poly::UnivariatePoly;

pub fn run(data: &[u8]) {
    let mut reader = Reader::new(data);
    match reader.u8() % 3 {
        0 => case::<fp128::Field, fp128::Field>(&mut reader),
        1 => case::<fp32::Field, fp32::ExtensionField>(&mut reader),
        _ => case::<fp64::Field, fp64::ExtensionField>(&mut reader),
    }
}

fn case<F, E>(reader: &mut Reader<'_>)
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + Fold,
{
    let num_vars = usize::from(reader.u8() % 13);
    let point = gen::point::<F, E>(reader, num_vars);
    let size = 1usize << num_vars;
    let lagrange: Vec<E> = (0..size)
        .map(|index| index_weight(&point, index, true))
        .collect();
    let monomial: Vec<E> = (0..size)
        .map(|index| index_weight(&point, index, false))
        .collect();

    assert_eq!(
        lagrange_weights(&point).expect("lagrange weights"),
        lagrange,
        "lagrange_weights"
    );
    assert_eq!(
        monomial_weights(&point).expect("monomial weights"),
        monomial,
        "monomial_weights"
    );
    assert_eq!(
        basis_weights(&point, BasisMode::Lagrange).expect("weights"),
        lagrange,
        "basis_weights(Lagrange)"
    );
    assert_eq!(
        basis_weights(&point, BasisMode::Monomial).expect("weights"),
        monomial,
        "basis_weights(Monomial)"
    );
    let live = reader.u32() as usize % (size + 1);
    for (basis, full) in [
        (BasisMode::Lagrange, &lagrange),
        (BasisMode::Monomial, &monomial),
    ] {
        let prefix = basis_weights_prefix(&point, basis, live);
        if live == 0 {
            assert!(
                matches!(prefix, Err(AkitaError::InvalidSize { .. })),
                "empty prefix is rejected"
            );
        } else {
            assert_eq!(
                prefix.expect("prefix weights"),
                full[..live],
                "basis_weights_prefix({basis:?}, {live})"
            );
        }
        assert!(
            matches!(
                basis_weights_prefix(&point, basis, size + 1),
                Err(AkitaError::InvalidSize { .. })
            ),
            "over-capacity prefix is rejected"
        );
    }

    let scale = gen::ext_scalar::<F, E>(reader);
    let scaled: Vec<E> = lagrange.iter().map(|&weight| weight * scale).collect();
    assert_eq!(
        EqPolynomial::evals(&point).expect("eq"),
        lagrange,
        "EqPolynomial::evals"
    );
    assert_eq!(
        EqPolynomial::evals_serial(&point, Some(scale)).expect("eq"),
        scaled,
        "evals_serial"
    );
    assert_eq!(
        EqPolynomial::evals_parallel(&point, Some(scale)).expect("eq"),
        scaled,
        "evals_parallel"
    );
    assert_eq!(
        EqPolynomial::evals_with_scaling(&point, Some(scale)).expect("eq"),
        scaled,
        "evals_with_scaling"
    );
    let mapped = EqPolynomial::evals_mapped(&point, |weight| weight + scale).expect("eq");
    assert_eq!(
        mapped,
        lagrange.iter().map(|&w| w + scale).collect::<Vec<_>>(),
        "evals_mapped"
    );
    // The implementation and its only caller (`GruenSplitEq`) use suffix
    // tables, `result[j] = eq(r[n-j..], ·)`; the doc comment says prefixes.
    // See `fuzz/FINDINGS.md`. The relied-upon behavior is asserted.
    let cached = EqPolynomial::evals_cached(&point).expect("eq");
    assert_eq!(cached.len(), num_vars + 1, "evals_cached depth");
    for (depth, table) in cached.iter().enumerate() {
        let suffix = &point[num_vars - depth..];
        let expected: Vec<E> = (0..1usize << depth)
            .map(|index| index_weight(suffix, index, true))
            .collect();
        assert_eq!(*table, expected, "evals_cached[{depth}]");
    }
    assert_eq!(
        EqPolynomial::evals_prefix(&point, live).expect("eq"),
        lagrange[..live],
        "evals_prefix"
    );
    assert_eq!(
        EqPolynomial::prefix_sum(&point, live).expect("eq"),
        lagrange[..live].iter().fold(E::zero(), |acc, &w| acc + w),
        "prefix_sum"
    );
    let split = SplitEqEvals::new(&point).expect("split eq");
    assert_eq!(split.len(), size, "SplitEqEvals::len");
    for index in [
        0,
        live.min(size - 1),
        size - 1,
        reader.u32() as usize % size,
    ] {
        assert_eq!(
            split.eval_at(index).expect("in range"),
            lagrange[index],
            "SplitEqEvals::eval_at"
        );
    }

    let other = gen::point::<F, E>(reader, num_vars);
    let other_weights: Vec<E> = (0..size)
        .map(|index| index_weight(&other, index, true))
        .collect();
    let inner = lagrange
        .iter()
        .zip(&other_weights)
        .fold(E::zero(), |acc, (&a, &b)| acc + a * b);
    assert_eq!(
        EqPolynomial::mle(&point, &other).expect("same length"),
        inner,
        "eq(x, y) = Σ_b eq(x, b) eq(y, b)"
    );

    let table: Vec<F> = gen::table(reader, size, gen::Domain::Full);
    let lifted: Vec<E> = table.iter().map(|&value| E::lift_base(value)).collect();
    assert_eq!(
        akita_algebra::poly::multilinear_eval(&lifted, &point).expect("matching length"),
        oracle::dense_lagrange(&table, &point),
        "multilinear_eval"
    );

    // Folding binds the first variable; an odd live prefix pads with zero.
    let live_len = 1 + reader.u32() as usize % size;
    let mut folded = lifted[..live_len].to_vec();
    let r = gen::ext_scalar::<F, E>(reader);
    akita_algebra::poly::fold_evals_in_place(&mut folded, r);
    let expected: Vec<E> = (0..live_len.div_ceil(2))
        .map(|pair| {
            let low = lifted[2 * pair];
            let high = if 2 * pair + 1 < live_len {
                lifted[2 * pair + 1]
            } else {
                E::zero()
            };
            low + r * (high - low)
        })
        .collect();
    assert_eq!(folded, expected, "fold_evals_in_place");

    if num_vars > 0 {
        gruen::<F, E>(reader, &point, scale);
    }
    stats::count("multilinear");
}

/// Round-by-round invariant of the split equality polynomial.
fn gruen<F, E>(reader: &mut Reader<'_>, tau: &[E], initial: E)
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    let mut split = GruenSplitEq::with_initial_scalar(tau, initial).expect("nonempty tau");
    let mut scalar = initial;
    for round in 0..tau.len() {
        assert_eq!(
            split.current_scalar(),
            scalar,
            "current scalar at round {round}"
        );
        assert_eq!(split.current_tau(), tau[round], "current tau");
        let (l0, l1) = split.linear_factor_evals();
        assert_eq!(l0, scalar * (E::one() - tau[round]), "l(0)");
        assert_eq!(l1, scalar * tau[round], "l(1)");
        let rest = &tau[round + 1..];
        let (first, second) = split.remaining_eq_tables();
        assert_eq!(
            first.len() * second.len(),
            1usize << rest.len(),
            "remaining table size"
        );
        for j in 0..1usize << rest.len() {
            let low = j & (first.len() - 1);
            let high = j >> first.len().trailing_zeros();
            assert_eq!(
                first[low] * second[high],
                index_weight(rest, j, true),
                "remaining eq table entry {j}"
            );
        }
        let coefficients: Vec<E> = (0..1 + usize::from(reader.u8() % 4))
            .map(|_| gen::ext_scalar::<F, E>(reader))
            .collect();
        let q = UnivariatePoly::new(coefficients);
        let x = gen::ext_scalar::<F, E>(reader);
        assert_eq!(
            split.gruen_mul(&q).evaluate(x),
            (l0 + x * (l1 - l0)) * q.evaluate(x),
            "gruen_mul is l(X) · q(X)"
        );
        let r = gen::ext_scalar::<F, E>(reader);
        split.bind(r);
        scalar *= tau[round] * r + (E::one() - tau[round]) * (E::one() - r);
    }
    assert_eq!(split.current_scalar(), scalar, "final scalar");
}
