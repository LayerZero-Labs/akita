#![cfg(feature = "labinius")]

mod common;

use akita_algebra::binary::{BinaryField128, BinaryField192};
use akita_challenges::BinaryChallengeSampler;
use akita_error::AkitaError;
use akita_labinius_prover::{
    fold_kernel::fold_integer,
    lowered::{encode_witness, prime_left_opening},
    prove_binary_clear_bytes,
};
use akita_labinius_verifier::{
    admitted::field_characteristic,
    channel::{self, RootSumcheckVerifierChannel},
    grinding::RootGrindingPlan,
    lowered::{
        coefficient_weights, prime_row_weights, LoweredChallenges, LoweredPrime, LoweredPublic,
        LoweredRootLayout,
    },
    root::{root_reduction_wire_size, verify_root_reduction_bytes, TransparentRootVerifierOracle},
    verify_binary_clear_bytes, PrimeClaim, RootOpeningMode, RootStatement,
};
use akita_transcript::ChallengeCodec;
use common::{admitted, Admitted, FixedDraw, RootCase, TestHost};
use jolt_field::{
    CanonicalEncoding, Ext2, ExtField, Field, Prime128Offset275, Prime32Offset99, Prime64Offset59,
};

type H = BinaryField128;
type P128 = Prime128Offset275;
type P64 = Prime64Offset59;

const MODES: [RootOpeningMode; 3] = [
    RootOpeningMode::Binary,
    RootOpeningMode::Prime,
    RootOpeningMode::Both,
];

/// The root reduction in every opening mode over the base field `F` and the
/// challenge field `E`, after the clear opening of the same binary claim.
fn complete<T: TestHost, F: Field + CanonicalEncoding, E: ExtField<F> + ChallengeCodec>() {
    for (log_fold_width, mode) in [1, 0]
        .into_iter()
        .flat_map(|log| MODES.map(|mode| (log, mode)))
    {
        let case = RootCase::<T, F, E>::new(admitted(log_fold_width, 0x31), mode);
        let setup = case.admitted.setup();
        if mode == RootOpeningMode::Binary {
            let clear = prove_binary_clear_bytes(
                setup,
                &case.source,
                &case.commitment,
                &case.point,
                case.value,
            )
            .unwrap();
            verify_binary_clear_bytes(setup, &case.commitment, &case.point, case.value, &clear)
                .unwrap();
        }
        let (proof, claims) = case.prove().unwrap();
        assert!(case.verify(&proof).unwrap() == claims);
        assert_eq!(claims.prime.is_some(), mode.has_prime());
        // The size excludes the transparent oracle's tables and counts every
        // nonce at its widest: two bytes for the fold response and the plan's
        // maximum for the proof of work.
        let regions = regions(&case, &proof);
        let total = |selected: fn(&str) -> bool| -> usize {
            regions
                .iter()
                .filter(|region| selected(region.0))
                .map(|region| region.2)
                .sum()
        };
        let (oracle, nonces) = (
            total(|name| name == "W" || name == "uP"),
            total(|name| name.ends_with("nonce")),
        );
        assert_eq!(
            proof.len() - oracle - nonces + 2 + plan(&case).nonce_max_bytes().unwrap(),
            root_reduction_wire_size::<T, F, E>(case.admitted.shape(), mode).unwrap(),
            "{mode:?}"
        );
    }
}

#[test]
fn every_opening_mode_verifies_for_both_hosts_and_field_pairs() {
    complete::<H, P128, P128>();
    complete::<BinaryField192, P128, P128>();
    complete::<H, P64, Ext2<P64>>();
    complete::<BinaryField192, P64, Ext2<P64>>();
}

fn plan<T: TestHost, F: Field + CanonicalEncoding, E: ExtField<F> + ChallengeCodec>(
    case: &RootCase<T, F, E>,
) -> RootGrindingPlan {
    RootGrindingPlan::new::<T, F, E>(case.admitted.shape(), case.layout.encoding(), case.mode)
        .unwrap()
}

/// Message regions of the case's mode in wire order; the transparent oracle's
/// public image is absent. A nonce is one canonical LEB128 value, so its
/// length is read from the proof. Both exercised challenge fields have fewer
/// than `2^128` elements, so every challenge-field site needs work and carries
/// a nonce.
fn regions<T: TestHost, F: Field + CanonicalEncoding, E: ExtField<F> + ChallengeCodec>(
    case: &RootCase<T, F, E>,
    proof: &[u8],
) -> Vec<(&'static str, usize, usize)> {
    let encoding = case.layout.encoding();
    let integers = |count: usize, bits: u32| count * (bits as usize).div_ceil(8);
    let field = E::DEGREE * F::NUM_BYTES;
    let frontend = T::ROWS * size_of::<T::Source>() + (2 * case.point.len() + 1) * 21;
    let (binary, prime) = (case.mode.has_binary(), case.mode.has_prime());
    // `None` is a nonce; the flag says whether the mode has the part.
    let head = [
        (binary, "frontend", Some(frontend)),
        (binary, "U", Some(case.layout.columns() * 21)),
        (prime, "uP", Some(case.layout.prime_len() * field)),
        (true, "fold nonce", None),
        (true, "W", Some(case.layout.witness_len())),
        (
            true,
            "KA",
            Some(integers(encoding.a_carry_len(), encoding.a_carry().bits())),
        ),
        (
            binary,
            "Q",
            Some(integers(
                encoding.parity_quotient_len(),
                encoding.quotient().bits(),
            )),
        ),
        (
            binary,
            "K",
            Some(integers(
                encoding.parity_carry_len(),
                encoding.carry().bits(),
            )),
        ),
        (true, "alpha nonce", None),
        (binary, "xi nonce", None),
        (true, "gamma nonce", None),
        (prime, "eta nonce", None),
        (true, "y_Y", Some(field)),
        (prime, "y_P", Some(field)),
    ];
    let mut parts: Vec<_> = head
        .into_iter()
        .filter(|part| part.0)
        .map(|(_, name, len)| (name, len))
        .collect();
    for (rounds, round, evaluation) in [
        (case.layout.witness_log_len(), "response round", "w_eval"),
        (case.layout.image_log_len(), "image round", "y_eval"),
    ] {
        parts.extend([("equality point nonce", None), ("batch nonce", None)]);
        for _ in 0..rounds {
            parts.extend([(round, Some(17 * field)), ("round nonce", None)]);
        }
        parts.push((evaluation, Some(field)));
    }
    if prime {
        parts.push(("batch nonce", None));
        for _ in 0..case.layout.prime_log_len() {
            parts.extend([("prime round", Some(2 * field)), ("round nonce", None)]);
        }
        parts.push(("p_eval", Some(field)));
    }
    let mut start = 0;
    parts
        .into_iter()
        .map(|(name, len)| {
            let nonce = || {
                1 + proof[start..]
                    .iter()
                    .take_while(|&&byte| byte >= 0x80)
                    .count()
            };
            let region = (name, start, len.unwrap_or_else(nonce));
            start += region.2;
            region
        })
        .collect()
}

fn tampered<F: Field + CanonicalEncoding, E: ExtField<F> + ChallengeCodec>(mode: RootOpeningMode) {
    let case = RootCase::<H, F, E>::new(admitted(1, 0x31), mode);
    let (binary, prime) = (mode.has_binary(), mode.has_prime());
    let (proof, _) = case.prove().unwrap();
    let regions = regions(&case, &proof);
    let (_, start, len) = *regions.last().unwrap();
    assert_eq!(start + len, proof.len());
    let rejects = |name: &str, result: Result<_, AkitaError>| {
        assert!(
            matches!(result, Err(AkitaError::InvalidProof)),
            "accepted {name}"
        );
    };

    // Proof bytes: a flipped bit in every message region, a noncanonical field
    // element, an integer with an unused wire bit set, and both length changes.
    // A flipped nonce is another value of its range, which fails its predicate
    // or selects another challenge.
    for &(name, start, len) in &regions {
        assert!(len > 0, "{name}");
        for position in [start, start + len - 1] {
            let mut changed = proof.clone();
            changed[position] ^= 1;
            rejects(name, case.verify(&changed));
        }
    }
    let region = |name: &str| *regions.iter().find(|region| region.0 == name).unwrap();
    // The characteristic itself is the least noncanonical coordinate encoding.
    let characteristic = field_characteristic::<F>().unwrap().to_le_bytes();
    let prime_elements = ["uP", "y_P", "prime round", "p_eval"];
    for name in ["y_Y", "response round", "w_eval", "image round", "y_eval"]
        .into_iter()
        .chain(prime_elements.into_iter().filter(|_| prime))
    {
        for coordinate in [0, E::DEGREE - 1] {
            let mut changed = proof.clone();
            changed[region(name).1 + coordinate * F::NUM_BYTES..][..F::NUM_BYTES]
                .copy_from_slice(&characteristic[..F::NUM_BYTES]);
            rejects(name, case.verify(&changed));
        }
    }
    let encoding = case.layout.encoding();
    for (name, bits) in [
        (true, "KA", encoding.a_carry().bits()),
        (binary, "Q", encoding.quotient().bits()),
        (binary, "K", encoding.carry().bits()),
    ]
    .into_iter()
    .filter_map(|(present, name, bits)| present.then_some((name, bits)))
    {
        if bits % 8 != 0 {
            let mut changed = proof.clone();
            changed[region(name).1 + (bits as usize).div_ceil(8) - 1] |= 1 << (bits % 8);
            rejects(name, case.verify(&changed));
        }
    }
    // 4096 is the first fold-response nonce outside the search domain.
    let (_, nonce_at, nonce_len) = region("fold nonce");
    let mut changed = proof.clone();
    changed.splice(nonce_at..nonce_at + nonce_len, [0x80, 0x20]);
    rejects("fold nonce", case.verify(&changed));
    // A proof-of-work site whose nonce is absent.
    let (_, nonce_at, nonce_len) = region("alpha nonce");
    let mut changed = proof.clone();
    changed.drain(nonce_at..nonce_at + nonce_len);
    rejects("missing nonce", case.verify(&changed));
    let mut extended = proof.clone();
    extended.push(0);
    rejects("truncated proof", case.verify(&proof[..proof.len() - 1]));
    rejects("extended proof", case.verify(&extended));

    // Statement: the mode, every part of each claim, the setup seed and the
    // image digit table are bound. The two image rows change one live digit
    // of the verifier's table: to a value outside the alphabet, and to
    // another digit of the alphabet, which encodes a different residue.
    let verify = |admitted: &Admitted, statement: RootStatement<'_, H, E>, image: &[u8]| {
        let mut oracle = TransparentRootVerifierOracle::<F, E>::new(image);
        verify_root_reduction_bytes::<H, F, E, _, _, _>(admitted, &statement, &mut oracle, &proof)
    };
    let honest = case.statement(mode);
    let mut wrong_point = case.point.clone();
    wrong_point[0] += H::ONE;
    let wrong = |mutate: fn(&mut PrimeClaim<E>)| {
        let mut claim = case.prime.clone();
        mutate(&mut claim);
        claim
    };
    let wrong_claims = [
        ("prime value", wrong(|claim| claim.value += E::one())),
        ("prime weight", wrong(|claim| claim.weights[0] += E::one())),
        ("ring point", wrong(|claim| claim.ring_point[0] += E::one())),
        (
            "column point",
            wrong(|claim| claim.column_point[0] += E::one()),
        ),
    ];
    let mut statements: Vec<_> = MODES
        .into_iter()
        .filter(|&other| other != mode)
        .map(|other| ("mode", case.statement(other)))
        .collect();
    if let Some((point, value)) = honest.binary {
        statements.extend(
            [
                ("value", (point, value + H::ONE)),
                ("point", (&wrong_point, value)),
            ]
            .map(|(name, claim)| {
                let binary = Some(claim);
                (name, RootStatement { binary, ..honest })
            }),
        );
    }
    if prime {
        statements.extend(wrong_claims.iter().map(|(name, claim)| {
            let prime = Some(claim);
            (*name, RootStatement { prime, ..honest })
        }));
    }
    for (name, statement) in statements {
        assert!(
            verify(&case.admitted, statement, &case.image).is_err(),
            "accepted {name}"
        );
    }
    let mut outside_alphabet = case.image.clone();
    outside_alphabet[0] = 16;
    let mut wrong_residue = case.image.clone();
    wrong_residue[0] = (wrong_residue[0] + 1) % 16;
    let other = admitted(1, 0x32);
    for (name, admitted, image) in [
        ("setup seed", &other, &case.image),
        (
            "image digit outside the alphabet",
            &case.admitted,
            &outside_alphabet,
        ),
        (
            "image digits of another residue",
            &case.admitted,
            &wrong_residue,
        ),
    ] {
        assert!(verify(admitted, honest, image).is_err(), "accepted {name}");
    }

    // The prover refuses a false value of either claim, and a commitment to
    // another source: its row residual is not divisible by the commitment
    // prime.
    let mut false_value = RootCase::<H, F, E>::new(admitted(1, 0x31), mode);
    false_value.value += H::ONE;
    assert_eq!(false_value.prove().is_err(), binary);
    let mut false_value = RootCase::<H, F, E>::new(admitted(1, 0x31), mode);
    false_value.prime.value += E::one();
    assert_eq!(false_value.prove().is_err(), prime);
    let mut other_source = RootCase::<H, F, E>::new(admitted(1, 0x31), mode);
    H::flip(&mut other_source.source[0]);
    other_source.value = common::host_mle::<H>(&other_source.source, &other_source.point);
    other_source.prime.value = common::cell_functional::<H, E>(
        &other_source.source,
        &other_source.cell_point,
        &other_source.bit_weights,
    );
    assert!(matches!(
        other_source.prove(),
        Err(AkitaError::InvalidProof)
    ));
}

/// The prime row at explicit challenges, against the attack it exists for:
/// opposite errors in two columns of the prime left opening that cancel in
/// the value claim. The production prover cannot prove a false row, because
/// its sumcheck refuses a false sum, so no rejecting proof exists to replay.
fn prime_row<F: Field + CanonicalEncoding, E: ExtField<F> + ChallengeCodec>() {
    let case = RootCase::<H, F, E>::new(admitted(1, 0x31), RootOpeningMode::Prime);
    let (setup, layout) = (case.admitted.setup(), &case.layout);
    let fold = BinaryChallengeSampler::new(setup.profile().clone())
        .sample_challenges(&mut FixedDraw, setup.columns())
        .unwrap();
    let response =
        fold_integer::<H>(&case.source, setup.scalar_rows(), 2, &fold, setup.profile()).unwrap();
    let digits = encode_witness(layout, &response).unwrap();
    // The stored response against its weights, minus the constant, is linear
    // in `eta`; the slope is the response side of the prime row.
    let response_side = |eta: u64| {
        let public = LoweredPublic::new(
            layout,
            setup,
            &fold,
            &vec![0; layout.encoding().a_carry_len()],
            LoweredChallenges {
                alpha: E::from_u64(0x9e37_79b9),
                gamma: E::from_u64(0x7f4a_7c15),
            },
            None,
            Some(LoweredPrime {
                ring_point: &case.prime.ring_point,
                eta: E::from_u64(eta),
            }),
        )
        .unwrap();
        let weights = coefficient_weights(layout, &public, setup).unwrap();
        let sum = digits
            .chunks_exact(public.digit_powers().len())
            .zip(&weights)
            .fold(E::zero(), |sum, (digits, &weight)| {
                let stored = digits.iter().zip(public.digit_powers());
                sum + weight
                    * stored.fold(E::zero(), |acc, (&d, &p)| acc + E::from_u64(d.into()) * p)
            });
        (sum - public.c_pub(), public)
    };
    let ((base, public), (with_row, _)) = (response_side(0), response_side(1));
    let row = prime_row_weights(layout, &public).unwrap();
    let value = case.prime.value_weights(layout).unwrap();
    let dot = |table: &[E], weights: &[E]| {
        let products = table.iter().zip(weights);
        products.fold(E::zero(), |sum, (&entry, &weight)| sum + entry * weight)
    };
    // The helper's weights against the functional read from the source words.
    let honest = prime_left_opening::<H, E>(layout, &case.source, &case.prime.ring_point).unwrap();
    assert!(dot(&honest, &value) == case.prime.value);
    assert!(dot(&honest, &row) == with_row - base);
    // Coefficient zero of column 0 moves by `eq(r_col, 1)` and of column 1 by
    // `-eq(r_col, 0)`.
    let r = case.prime.column_point[0];
    let mut shifted = honest;
    shifted[0] += r;
    shifted[layout.padded_coefficients()] -= E::one() - r;
    assert!(dot(&shifted, &value) == case.prime.value);
    assert!(dot(&shifted, &row) != with_row - base);
}

#[test]
fn prime_row_rejects_column_errors_that_cancel_in_the_value_claim() {
    prime_row::<P128, P128>();
    prime_row::<P64, Ext2<P64>>();
}

#[test]
fn tampered_reductions_and_statements_reject_in_every_mode_on_both_field_pairs() {
    for mode in MODES {
        tampered::<P128, P128>(mode);
        tampered::<P64, Ext2<P64>>(mode);
    }
    // A base field that fails proof-prime admission cannot build a layout, and
    // neither can a challenge field too small for the reduction's challenges.
    let admitted = admitted(1, 0x31);
    let (setup, shape) = (admitted.setup(), admitted.shape());
    for layout in [
        LoweredRootLayout::new::<Prime32Offset99, Prime32Offset99, _, _>(setup, shape),
        LoweredRootLayout::new::<P64, P64, _, _>(setup, shape),
    ] {
        assert!(matches!(layout, Err(AkitaError::InvalidSetup(_))));
    }
}

/// A native borrowed reduction consumes only its own messages. Every opening
/// mode replays on both shipped field pairs and leaves a following message for
/// the surrounding protocol.
fn borrowed_root<F: Field + CanonicalEncoding, E: ExtField<F> + ChallengeCodec>() {
    use akita_labinius_prover::{prove_root_reduction_in_session, TransparentRootProverOracle};
    use akita_labinius_verifier::root::verify_root_reduction_in_session;
    use akita_transcript::{new_prover_channel, new_verifier_channel, ProverTranscript};
    use jolt_transcript::BytesMsg;

    for mode in MODES {
        let case = RootCase::<H, F, E>::new(admitted(1, 0x31), mode);
        let (standalone, _) = case.prove().unwrap();
        case.verify(&standalone).unwrap();
        let prove = |prefix: &[u8]| {
            let mut session = new_prover_channel(b"root-composition-test");
            session.public_message(&BytesMsg(prefix));
            let claims = prove_root_reduction_in_session::<H, F, E, _, _, _>(
                &case.admitted,
                &case.source,
                &case.commitment,
                &case.statement(mode),
                &mut TransparentRootProverOracle::<F, E>::new(&case.image),
                &mut session,
            )
            .unwrap();
            session.public_message(&BytesMsg(b"suffix"));
            let final_challenge: [u8; 32] = session.verifier_message();
            ProverTranscript::send(&mut session, &BytesMsg(final_challenge.to_vec()));
            (session.narg_string().to_vec(), claims, final_challenge)
        };
        let (proof, claims, challenge) = prove(b"prefix");
        let (_, other_claims, _) = prove(b"changed prefix");
        assert_ne!(claims.response_point, other_claims.response_point);
        let replay = |proof: &[u8], session: &[u8], prefix: &[u8], suffix: &[u8]| {
            let mut state = new_verifier_channel(session, proof);
            state.public_message(&BytesMsg(prefix));
            let actual = verify_root_reduction_in_session::<H, F, E, _, _, _>(
                &case.admitted,
                &case.statement(mode),
                &mut TransparentRootVerifierOracle::<F, E>::new(&case.image),
                &mut state,
            )?;
            assert_eq!(actual, claims);
            state.public_message(&BytesMsg(suffix));
            let expected: [u8; 32] = state.verifier_message();
            let received = state
                .prover_message::<BytesMsg>()
                .map_err(|_| AkitaError::InvalidProof)?;
            if received.0 != expected || expected != challenge {
                return Err(AkitaError::InvalidProof);
            }
            state.check_eof().map_err(|_| AkitaError::InvalidProof)
        };
        let valid = |bytes: &[u8]| replay(bytes, b"root-composition-test", b"prefix", b"suffix");
        valid(&proof).unwrap();
        for (session, prefix, suffix) in [
            (
                b"wrong session".as_slice(),
                b"prefix".as_slice(),
                b"suffix".as_slice(),
            ),
            (
                b"root-composition-test".as_slice(),
                b"wrong prefix".as_slice(),
                b"suffix".as_slice(),
            ),
            (
                b"root-composition-test".as_slice(),
                b"prefix".as_slice(),
                b"wrong suffix".as_slice(),
            ),
        ] {
            assert!(replay(&proof, session, prefix, suffix).is_err());
        }
        // Replay the same reduction under a mismatched root identity, while
        // keeping the outer session and its prefix identical.
        let mut identity = case.admitted.identity_bytes::<H>().unwrap();
        identity[0] ^= 1;
        let mut state = new_verifier_channel(b"root-composition-test", &proof);
        state.public_message(&BytesMsg(b"prefix"));
        channel::bind_root_context(&mut state, &identity).unwrap();
        assert!(
            akita_labinius_verifier::root::verify_root_reduction::<H, F, E, _, _, _, _>(
                &case.admitted,
                &case.statement(mode),
                &mut TransparentRootVerifierOracle::<F, E>::new(&case.image),
                &mut RootSumcheckVerifierChannel::<F>::new(&mut state),
            )
            .is_err()
        );
        assert!(valid(&proof[..proof.len() - 33]).is_err());
        for name in ["fold nonce", "alpha nonce", "response round"] {
            let (_, start, _) = regions(&case, &proof)
                .into_iter()
                .find(|region| region.0 == name)
                .unwrap();
            let mut corrupted = proof.clone();
            corrupted[start] ^= 1;
            assert!(valid(&corrupted).is_err(), "accepted corrupt {name}");
        }
    }
}

#[test]
fn borrowed_root_reductions_replay_and_leave_the_session_open() {
    borrowed_root::<P128, P128>();
    borrowed_root::<P64, Ext2<P64>>();
}
