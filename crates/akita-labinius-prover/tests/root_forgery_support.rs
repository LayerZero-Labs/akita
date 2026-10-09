#![cfg(feature = "labinius")]
#![allow(dead_code)]

#[path = "root_reduction_support.rs"]
pub(crate) mod root_reduction_support;

use self::root_reduction_support::{Case, F, H};
use akita_algebra::MinusTrinomial;
use akita_algebra::{embed_scalar, TrinomialRing};
use akita_challenges::BinaryChallengeSampler;
use akita_labinius_prover::{
    commit_binary_clear,
    lowered::{encode_witness, flatten_image},
};
use akita_labinius_verifier::{
    channel::{
        self, ClearChannel, RootChallengeChannel, RootFieldSite, RootSumcheckProverChannel,
        RootSumcheckVerifierChannel,
    },
    codec::{exchange_binary, exchange_field},
    endpoint::{fold_integer, left_expansion, pack_response, verify_left_expansion},
    frontend::{prove_frontend, verify_frontend},
    lowered::{image_weights_dense, witness_weights_dense, LoweredChallenges, LoweredPublic},
    root::{
        bind_root_statement, exchange_root_auxiliary, RootVerifierOracle,
        TransparentRootVerifierOracle,
    },
    root_sumcheck::{
        alphabet_polynomial, bind_root_sumcheck_instance, combined_shape, combined_terminal,
        product_shape, verify_combined_rounds, RootSumcheckInstance,
    },
    source::{challenge_scalar, equality_weights},
};
use akita_params::sis::labinius::LabiniusDigitBase;
use akita_sumcheck::{prove_sumcheck, InfallibleSumcheck, SumcheckInstanceProver};
use jolt_field::{Field, One, Ring, Zero};
use jolt_poly::UnivariatePoly;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Attack {
    Honest,
    BadDigit,
    NonzeroTail,
    ParityOnly,
    AOnly,
    WrongY,
}

/// A malicious driver keeps every round consistent with its claimed sum by
/// shifting the constant; its actual table evaluations expose that shift.
struct Forged {
    w: Vec<F>,
    kw: Vec<F>,
    eq: Vec<F>,
    base: Option<LabiniusDigitBase>,
    beta: F,
    claim: F,
    rounds: usize,
}
impl Forged {
    fn fold(table: &mut Vec<F>, challenge: F) {
        for i in 0..table.len() / 2 {
            table[i] = table[2 * i] + challenge * (table[2 * i + 1] - table[2 * i]);
        }
        table.truncate(table.len() / 2);
    }
}
impl SumcheckInstanceProver<F> for Forged {
    fn num_rounds(&self) -> usize {
        self.rounds
    }
    fn degree_bound(&self) -> usize {
        self.base.map_or(2, |b| (1usize << b.bits()) + 1)
    }
    fn input_claim(&self) -> F {
        self.claim
    }
    fn compute_round_univariate(&mut self, _: usize, claim: F) -> UnivariatePoly<F> {
        let values: Vec<F> = (0..=self.degree_bound())
            .map(|node| {
                let t = F::from_u64(node as u64);
                self.w
                    .chunks_exact(2)
                    .zip(self.kw.chunks_exact(2))
                    .zip(self.eq.chunks_exact(2))
                    .fold(F::zero(), |sum, ((w, k), e)| {
                        let w = w[0] + t * (w[1] - w[0]);
                        let k = k[0] + t * (k[1] - k[0]);
                        let e = e[0] + t * (e[1] - e[0]);
                        sum + self
                            .base
                            .map_or(F::zero(), |b| e * alphabet_polynomial(b, w))
                            + self.beta * w * k
                    })
            })
            .collect();
        let shift = (claim - values[0] - values[1]) * F::from_u64(2).inverse().unwrap();
        UnivariatePoly::from_evals(&values.into_iter().map(|v| v + shift).collect::<Vec<_>>())
    }
    fn ingest_challenge(&mut self, _: usize, challenge: F) {
        Self::fold(&mut self.w, challenge);
        Self::fold(&mut self.kw, challenge);
        Self::fold(&mut self.eq, challenge);
    }
}

pub(crate) struct Evidence {
    pub(crate) proof: Vec<u8>,
    pub(crate) public: LoweredPublic<F>,
    pub(crate) response: Vec<[i64; 162]>,
    pub(crate) digits: Vec<u8>,
}

pub(crate) fn assemble(case: &mut Case, attack: Attack) -> Evidence {
    let setup = case.admitted.setup();
    let mut response_source = case.source.clone();
    if attack == Attack::AOnly {
        response_source[0] ^= 1;
        case.commitment =
            commit_binary_clear::<H, F, 648, MinusTrinomial>(setup, &response_source).unwrap();
        case.image = flatten_image(&case.layout, &case.commitment).unwrap();
    }
    let mut state = channel::new_root_prover().unwrap();
    let mut ch = RootSumcheckProverChannel::new(&mut state);
    bind_root_statement(
        &case.admitted,
        case.base,
        &case.point,
        case.value,
        &mut ch,
        |layout, ch| akita_labinius_verifier::root::bind_transparent_image(layout, &case.image, ch),
    )
    .unwrap();
    let binary = prove_frontend::<H, _>(&case.source, &case.point, case.value, &mut ch).unwrap();
    let mut u = left_expansion::<H>(
        &case.source,
        &binary.point,
        setup.scalar_rows(),
        setup.columns(),
    )
    .unwrap();
    for element in &mut u {
        exchange_binary(&mut ch, element).unwrap();
    }
    verify_left_expansion(setup, &binary, &u).unwrap();
    let fold = ch
        .fold_challenges(
            &mut BinaryChallengeSampler::new(setup.profile().clone()),
            b"akita/labinius/root-fold/v1",
            setup.columns(),
        )
        .unwrap();
    let mut response = fold_integer::<H>(
        &response_source,
        setup.scalar_rows(),
        setup.columns(),
        &fold,
        setup.profile(),
    )
    .unwrap();
    if attack == Attack::ParityOnly {
        response[0][0] += 2;
    }
    let mut digits = encode_witness(&case.layout, &response).unwrap();
    if attack == Attack::BadDigit {
        digits[648 * case.layout.encoding().response().digit_count()] = 255;
    }
    if attack == Attack::NonzeroTail {
        digits[648 * case.layout.encoding().response().digit_count()] = (1 << case.base.bits()) - 1;
    }
    // The commitment fixes raw bytes, including any chosen padding digit.
    ch.message(&mut digits).unwrap();
    let packed = pack_response(setup, &response).unwrap();
    let embedded: Vec<_> = fold
        .iter()
        .map(|c| {
            embed_scalar::<F, 162, 648, MinusTrinomial>(&challenge_scalar(c).unwrap()).unwrap()
        })
        .collect();
    let mut qa = Vec::new();
    for row in 0..setup.n_a() {
        let mut residual = vec![F::zero(); 1295];
        for (a, v) in setup.matrix()[row * setup.m()..(row + 1) * setup.m()]
            .iter()
            .zip(&packed)
        {
            for (r, c) in residual
                .iter_mut()
                .zip(a.schoolbook_product_coefficients(v).unwrap())
            {
                *r += c;
            }
        }
        for (col, c) in embedded.iter().enumerate() {
            for (r, v) in residual.iter_mut().zip(
                c.schoolbook_product_coefficients(&case.commitment.images[col * setup.n_a() + row])
                    .unwrap(),
            ) {
                *r -= v;
            }
        }
        let (_, quotient) =
            TrinomialRing::<F, 648, MinusTrinomial>::reduce_product_with_quotient(&residual)
                .unwrap();
        qa.push(quotient);
    }
    // Best integral carry even for a false parity row: long-divide the true
    // residual, then round odd remainder coefficients down instead of hiding it.
    let weights = equality_weights(&binary.point[..setup.row_vars()]).unwrap();
    let mut residual = [0i128; 323];
    for (weight, row) in weights.iter().zip(&response) {
        let bits = weight.to_bytes();
        for bit in 0..162 {
            if bits[bit / 8] & (1 << (bit % 8)) != 0 {
                for (j, &v) in row.iter().enumerate() {
                    residual[bit + j] += i128::from(v);
                }
            }
        }
    }
    for (u, c) in u.iter().zip(&fold) {
        let bits = u.to_bytes();
        for bit in 0..162 {
            if bits[bit / 8] & (1 << (bit % 8)) != 0 {
                for term in c.terms() {
                    residual[bit + usize::from(term.position)] -= i128::from(term.coefficient);
                }
            }
        }
    }
    let mut q = vec![0i128; 161];
    for degree in (162..323).rev() {
        let leading = residual[degree];
        q[degree - 162] = leading;
        residual[degree] = 0;
        residual[degree - 162] -= leading;
        residual[degree - 81] -= leading;
    }
    let mut k: Vec<_> = residual[..162].iter().map(|v| v.div_euclid(2)).collect();
    exchange_root_auxiliary(&case.layout, &mut ch, &mut qa, &mut q, &mut k).unwrap();
    let challenges = LoweredChallenges {
        alpha: ch.field_challenge(RootFieldSite::Alpha).unwrap(),
        xi: ch.field_challenge(RootFieldSite::Xi).unwrap(),
        gamma: ch.field_challenge(RootFieldSite::Gamma).unwrap(),
    };
    let public = LoweredPublic::new(
        &case.layout,
        setup,
        &binary,
        &u,
        &fold,
        &qa,
        &q,
        &k,
        challenges,
    )
    .unwrap();
    let ky = image_weights_dense(&case.layout, &public).unwrap();
    let mut y_y = case
        .image
        .iter()
        .zip(&ky)
        .fold(F::zero(), |s, (&y, &k)| s + y * k);
    if attack == Attack::WrongY {
        y_y += F::one();
    }
    exchange_field(&mut ch, &mut y_y).unwrap();
    let s = public.c_pub() - y_y;
    let tau: Vec<F> = (0..case.layout.witness_log_len())
        .map(|i| ch.field_challenge(RootFieldSite::Tau(i as u32)).unwrap())
        .collect();
    let beta: F = ch.field_challenge(RootFieldSite::Beta).unwrap();
    let eq = (0..digits.len())
        .map(|i| {
            tau.iter().enumerate().fold(F::one(), |s, (j, &t)| {
                s * if i >> j & 1 == 0 { F::one() - t } else { t }
            })
        })
        .collect();
    let kw = witness_weights_dense(&case.layout, &public).unwrap();
    let mut forged = Forged {
        w: digits.iter().map(|&d| F::from_u64(u64::from(d))).collect(),
        kw,
        eq,
        base: Some(case.base),
        beta,
        claim: beta * s,
        rounds: tau.len(),
    };
    bind_root_sumcheck_instance(
        &mut ch,
        RootSumcheckInstance::Combined(case.base),
        0,
        tau.len(),
    )
    .unwrap();
    prove_sumcheck::<F, F, _, _>(
        &mut InfallibleSumcheck(&mut forged),
        &mut ch,
        combined_shape(tau.len(), case.base).unwrap(),
        0,
    )
    .unwrap();
    let mut w_eval = forged.w[0];
    exchange_field(&mut ch, &mut w_eval).unwrap();
    let mut product = Forged {
        w: case.image.clone(),
        kw: ky,
        eq: vec![F::zero(); case.image.len()],
        base: None,
        beta: F::one(),
        claim: y_y,
        rounds: case.layout.image_log_len(),
    };
    bind_root_sumcheck_instance(&mut ch, RootSumcheckInstance::Product, 1, product.rounds).unwrap();
    let rounds = product.rounds;
    prove_sumcheck::<F, F, _, _>(
        &mut InfallibleSumcheck(&mut product),
        &mut ch,
        product_shape(rounds).unwrap(),
        1,
    )
    .unwrap();
    let mut y_eval = product.w[0];
    exchange_field(&mut ch, &mut y_eval).unwrap();
    Evidence {
        proof: channel::finish_prover(state),
        public,
        response,
        digits,
    }
}

/// Independently replay all mandatory checks before the combined terminal.
/// A false result identifies that terminal, rather than an earlier parser,
/// frontend, expansion, public-range or instance-binding check, as rejection.
pub(crate) fn combined_terminal_matches(case: &Case, proof: &[u8]) -> bool {
    let setup = case.admitted.setup();
    let mut state = channel::new_root_verifier(proof).unwrap();
    let mut ch = RootSumcheckVerifierChannel::new(&mut state);
    let mut oracle = TransparentRootVerifierOracle::new(&case.image);
    bind_root_statement(
        &case.admitted,
        case.base,
        &case.point,
        case.value,
        &mut ch,
        |l, c| oracle.bind_image(l, c),
    )
    .unwrap();
    let binary = verify_frontend(&case.point, case.value, &mut ch).unwrap();
    let mut u = vec![akita_algebra::binary::BinaryField162::ZERO; setup.columns()];
    for element in &mut u {
        exchange_binary(&mut ch, element).unwrap();
    }
    verify_left_expansion(setup, &binary, &u).unwrap();
    let fold = ch
        .fold_challenges(
            &mut BinaryChallengeSampler::new(setup.profile().clone()),
            b"akita/labinius/root-fold/v1",
            setup.columns(),
        )
        .unwrap();
    oracle.bind_response(&case.layout, &mut ch).unwrap();
    let mut qa = vec![vec![F::zero(); 647]; setup.n_a()];
    let mut q = vec![0; 161];
    let mut k = vec![0; 162];
    exchange_root_auxiliary(&case.layout, &mut ch, &mut qa, &mut q, &mut k).unwrap();
    let c = LoweredChallenges {
        alpha: ch.field_challenge(RootFieldSite::Alpha).unwrap(),
        xi: ch.field_challenge(RootFieldSite::Xi).unwrap(),
        gamma: ch.field_challenge(RootFieldSite::Gamma).unwrap(),
    };
    let public =
        LoweredPublic::new(&case.layout, setup, &binary, &u, &fold, &qa, &q, &k, c).unwrap();
    let mut y = F::zero();
    exchange_field(&mut ch, &mut y).unwrap();
    let tau: Vec<F> = (0..case.layout.witness_log_len())
        .map(|i| ch.field_challenge(RootFieldSite::Tau(i as u32)).unwrap())
        .collect();
    let beta = ch.field_challenge(RootFieldSite::Beta).unwrap();
    let result =
        verify_combined_rounds(&mut ch, 0, tau.len(), case.base, beta, public.c_pub() - y).unwrap();
    let mut w = F::zero();
    exchange_field(&mut ch, &mut w).unwrap();
    let kw = akita_labinius_verifier::lowered::witness_weight_mle(
        &case.layout,
        &public,
        setup,
        &result.challenges,
    )
    .unwrap();
    result.output_claim
        == combined_terminal(case.base, &tau, &result.challenges, beta, w, kw).unwrap()
}
