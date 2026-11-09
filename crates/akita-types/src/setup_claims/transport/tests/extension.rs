use super::super::*;
use crate::{derive_public_matrix_prefix, AkitaExpandedSetup, AkitaSetupDescriptor, FpExtEncoding};
use jolt_field::{Ext2, FpExt4, FpExt8, Prime128Offset6A23801F, Prime32Offset99, Prime64Offset59};

fn round_trip<F, E>()
where
    F: Field + CanonicalEncoding + ExtField<F>,
    E: FpExtEncoding<F>,
{
    let setup = AkitaExpandedSetup::<F>::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupDescriptor {
            max_num_vars: 6,
            max_num_batched_polys: 1,
            num_field_elements: 64,
            setup_seed: [19; 32].into(),
        },
        derive_public_matrix_prefix(64, &[19; 32].into()),
    );
    let polynomial = SetupPolynomialId::new::<F, E>(&setup, 64).unwrap();
    let value = |seed| E::from_base_fn(|coordinate| F::from_u64(seed + 13 * coordinate as u64));
    let native = SetupQuery::from_native_terms(vec![NativeTerm {
        coefficient: vec![value(2), value(3)],
        point: (4..8).map(value).collect(),
        tensors: vec![
            EqPairTensorFamily::new(0, 0, value(9), vec![EqPairTensorAxis::unit(4, 1, 1)]).unwrap(),
            EqPairTensorFamily::new(
                4,
                0,
                value(10),
                vec![EqPairTensorAxis::dense(
                    1,
                    1,
                    (11..15).map(value).collect::<Vec<_>>(),
                )],
            )
            .unwrap(),
            EqPairTensorFamily::new(
                8,
                0,
                value(15),
                vec![EqPairTensorAxis::bit_product(
                    1,
                    2,
                    vec![[value(16), value(17)], [value(18), value(19)]],
                )
                .unwrap()],
            )
            .unwrap(),
        ],
    }]);
    let claims = vec![
        SetupClaim::new(
            polynomial.clone(),
            SetupQuery::evaluation((20..26).map(value).collect()).unwrap(),
            value(26),
        )
        .unwrap(),
        SetupClaim::new(polynomial.clone(), native, value(27)).unwrap(),
    ];
    let statement = SetupBatchStatement::new(polynomial.clone(), 2, claims).unwrap();
    let bytes = statement.descriptor_bytes::<F>().unwrap();
    let decoded =
        SetupBatchStatement::<E>::from_descriptor_bytes::<F>(&bytes, &polynomial).unwrap();
    assert_eq!(decoded.descriptor_bytes::<F>().unwrap(), bytes);
    let point = (30..36).map(value).collect::<Vec<_>>();
    for (expected, actual) in statement.claims().iter().zip(decoded.claims()) {
        assert_eq!(expected.value(), actual.value());
        let terms = expected.query().compile(&polynomial, 2, None).unwrap();
        let dense = (0..64)
            .map(|flat| {
                terms.iter().fold(E::zero(), |sum, term| {
                    sum + term.coefficient()[flat % 2] * term.index_weight(flat / 2)
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(
            actual.query().evaluate_weight(&polynomial, &point).unwrap(),
            akita_algebra::poly::multilinear_eval(&dense, &point).unwrap()
        );
    }
    for end in 0..bytes.len() {
        assert!(
            SetupBatchStatement::<E>::from_descriptor_bytes::<F>(&bytes[..end], &polynomial)
                .is_err()
        );
    }
    // Every coordinate of the final extension answer must be canonical.
    for coordinate in 0..E::DEGREE {
        let mut malformed = bytes.clone();
        let start = malformed.len() - (E::DEGREE - coordinate) * F::NUM_BYTES;
        malformed[start..start + F::NUM_BYTES].fill(255);
        assert!(
            SetupBatchStatement::<E>::from_descriptor_bytes::<F>(&malformed, &polynomial).is_err()
        );
    }
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(SetupBatchStatement::<E>::from_descriptor_bytes::<F>(&trailing, &polynomial).is_err());
    let wrong_domain = SetupPolynomialId::new::<F, E>(&setup, 32).unwrap();
    assert!(SetupBatchStatement::<E>::from_descriptor_bytes::<F>(&bytes, &wrong_domain).is_err());
    let empty = SetupBatchStatement::<E>::new(polynomial.clone(), 2, Vec::new()).unwrap();
    let empty_bytes = empty.descriptor_bytes::<F>().unwrap();
    assert!(
        SetupBatchStatement::<E>::from_descriptor_bytes::<F>(&empty_bytes, &polynomial)
            .unwrap()
            .claims()
            .is_empty()
    );
    if E::DEGREE > 1 {
        assert!(
            SetupBatchStatement::<F>::from_descriptor_bytes::<F>(&empty_bytes, &polynomial)
                .is_err()
        );
    }
}

#[test]
fn canonical_transport_preserves_base_field_encoding() {
    round_trip::<Prime128Offset6A23801F, Prime128Offset6A23801F>();
}

#[test]
fn canonical_transport_round_trips_genuine_quadratic_values() {
    round_trip::<Prime64Offset59, Ext2<Prime64Offset59>>();
}

#[test]
fn canonical_transport_round_trips_genuine_quartic_values() {
    round_trip::<Prime32Offset99, FpExt4<Prime32Offset99>>();
}

#[test]
fn canonical_transport_round_trips_genuine_octic_values() {
    round_trip::<Prime32Offset99, FpExt8<Prime32Offset99>>();
}

fn native_dimension<F, E>(dimension: usize, accepted: bool)
where
    F: Field + CanonicalEncoding,
    E: FpExtEncoding<F>,
{
    let fields = 2 * dimension;
    let setup = AkitaExpandedSetup::<F>::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupDescriptor {
            max_num_vars: fields.trailing_zeros() as usize,
            max_num_batched_polys: 1,
            num_field_elements: fields,
            setup_seed: [19; 32].into(),
        },
        derive_public_matrix_prefix(fields, &[19; 32].into()),
    );
    let polynomial = SetupPolynomialId::new::<F, E>(&setup, fields).unwrap();
    let query = SetupQuery::from_native_terms(vec![NativeTerm {
        coefficient: (0..dimension)
            .map(|i| E::from_base_fn(|coordinate| F::from_u64((i + coordinate + 3) as u64)))
            .collect(),
        point: vec![E::one()],
        tensors: vec![EqPairTensorFamily::new(
            0,
            0,
            E::one(),
            vec![EqPairTensorAxis::unit(2, 1, 1)],
        )
        .unwrap()],
    }]);
    let statement = SetupBatchStatement::new(
        polynomial.clone(),
        2,
        vec![SetupClaim::new(polynomial.clone(), query, E::one()).unwrap()],
    )
    .unwrap();
    let bytes = statement.descriptor_bytes::<F>().unwrap();
    let decoded = SetupBatchStatement::<E>::from_descriptor_bytes::<F>(&bytes, &polynomial);
    assert_eq!(decoded.is_ok(), accepted, "native dimension {dimension}");
    if let Ok(decoded) = decoded {
        assert_eq!(decoded.descriptor_bytes::<F>().unwrap(), bytes);
    }
}

#[test]
fn canonical_transport_bounds_native_dimensions_by_field_dispatch() {
    native_dimension::<Prime128Offset6A23801F, Prime128Offset6A23801F>(1024, true);
    native_dimension::<Prime128Offset6A23801F, Prime128Offset6A23801F>(2048, false);
    native_dimension::<Prime64Offset59, Ext2<Prime64Offset59>>(2048, true);
    native_dimension::<Prime64Offset59, Ext2<Prime64Offset59>>(4096, false);
    native_dimension::<Prime32Offset99, FpExt4<Prime32Offset99>>(2048, true);
    native_dimension::<Prime32Offset99, FpExt4<Prime32Offset99>>(4096, false);
}
