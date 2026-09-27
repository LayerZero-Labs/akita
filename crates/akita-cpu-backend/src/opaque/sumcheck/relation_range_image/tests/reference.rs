use super::trace_prefix::two_source_linear_terms;
use super::*;

#[derive(Clone, Copy, Debug)]
enum WeightMode {
    Prefix,
    Disabled,
    ReducedDense,
}

/// Dense Boolean tables, folded independently of the prover's phase machinery.
struct Reference {
    witness: Vec<F>,
    linear: Vec<F>,
    range: Vec<F>,
}

impl Reference {
    fn claim(&self) -> F {
        (0..self.witness.len())
            .map(|i| {
                let w = self.witness[i];
                w * self.linear[i] + self.range[i] * w * (w + F::one())
            })
            .sum()
    }

    fn round(&self) -> UnivariatePoly<F> {
        let evals = [0, 1, 2, 3].map(|x| {
            let x = F::from_u64(x);
            (0..self.witness.len() / 2)
                .map(|i| {
                    let at = |table: &[F]| table[2 * i] + x * (table[2 * i + 1] - table[2 * i]);
                    let w = at(&self.witness);
                    w * at(&self.linear) + at(&self.range) * w * (w + F::one())
                })
                .sum::<F>()
        });
        let mut poly = UnivariatePoly::from_evals(&evals);
        poly.trim_trailing_zeros();
        poly
    }

    fn bind(&mut self, challenge: F) {
        for table in [&mut self.witness, &mut self.linear, &mut self.range] {
            *table = table
                .chunks_exact(2)
                .map(|pair| pair[0] + challenge * (pair[1] - pair[0]))
                .collect();
        }
    }
}

fn check_configuration(coefficient_bits: usize, live_lanes: usize, mode: WeightMode, sparse: bool) {
    let lane_bits = 5;
    let coeff_count = 1 << coefficient_bits;
    let domain_len = coeff_count << lane_bits;
    let num_vars = coefficient_bits + lane_bits;
    let label =
        format!("bits={coefficient_bits}, live={live_lanes}, mode={mode:?}, sparse={sparse}");
    let witness = (0..live_lanes * coeff_count)
        .map(|i| ((i * 5 + i / 7 + 3) % 8) as i8 - 4)
        .collect::<Vec<_>>();
    let point = (0..num_vars)
        .map(|i| F::from_u64(17 + 3 * i as u64))
        .collect::<Vec<_>>();
    let gamma = F::from_u64(53);
    let common = (0..coeff_count)
        .map(|i| F::from_u64(29 + 5 * i as u64))
        .collect::<Vec<_>>();
    let lanes = (0..1 << lane_bits)
        .map(|i| F::from_u64(41 + 7 * i as u64))
        .collect::<Vec<_>>();
    let (structured, mut linear) = two_source_linear_terms(live_lanes, coeff_count);
    let params = Stage2Params {
        stage1_point: &point,
        b: 8,
        live_lane_count: live_lanes,
        lane_bits,
        coefficient_bits,
    };
    let direct =
        direct_relation_range_image_evaluation(gamma, &witness, &common, &lanes, &linear, &params);
    linear.resize(domain_len, F::zero());
    let relation = (0..domain_len)
        .map(|i| {
            let factored = common[i % coeff_count] * lanes[i / coeff_count];
            if matches!(mode, WeightMode::ReducedDense) {
                factored + F::from_u64((i * i % 97) as u64)
            } else {
                factored
            }
        })
        .collect::<Vec<_>>();
    let relation_claim = witness
        .iter()
        .zip(&relation)
        .map(|(&w, &p)| F::from_i64(i64::from(w)) * p)
        .sum();
    for (total, &weight) in linear.iter_mut().zip(&relation) {
        *total += weight;
    }
    let oracle = match mode {
        WeightMode::ReducedDense => RelationWeightOracle::ReducedDense(
            DenseRelationWeights::new(relation, witness.len()).unwrap(),
        ),
        _ => RelationWeightOracle::QuotientFactored(
            RelationWeightFactorization::new(common, lanes).unwrap(),
        ),
    };
    let mut range = EqPolynomial::evals(&point).unwrap();
    for weight in &mut range {
        *weight *= gamma;
    }
    let packed_witness = packed(&witness);
    let additional = sparse.then(|| {
        // Include support in the padded tail: it contributes after lane folding.
        let weights = [1, coeff_count + 1, witness.len() - 1, domain_len - 1]
            .map(|i| (i, F::from_u64(61 + i as u64)));
        for &(i, weight) in &weights {
            linear[i] += weight;
        }
        let intervals = [
            1..3,
            witness.len() - 1..witness.len() + usize::from(witness.len() < domain_len),
        ];
        let binary_point = (0..num_vars)
            .map(|i| F::from_u64(71 + 11 * i as u64))
            .collect::<Vec<_>>();
        let binary_eq = EqPolynomial::evals(&binary_point).unwrap();
        let rho = F::from_u64(83);
        for interval in &intervals {
            for i in interval.clone() {
                range[i] += rho * binary_eq[i];
            }
        }
        AdditionalRelationTerms::new(
            &packed_witness,
            domain_len,
            weights.to_vec(),
            &intervals,
            &binary_point,
            rho,
        )
        .unwrap()
    });
    let mut prover = RelationRangeImageProver::new(
        gamma,
        packed_witness,
        &point,
        direct.range_image,
        8,
        oracle,
        live_lanes,
        lane_bits,
        coefficient_bits,
        relation_claim,
        structured,
        direct.evaluation_trace,
        additional,
    )
    .unwrap();
    if matches!(mode, WeightMode::Disabled) {
        prover.disable_compact_quotient_prefix();
    }
    // One coefficient bit intentionally uses the ordinary compact first round.
    assert_eq!(
        prover.compact_quotient_prefix().is_some(),
        matches!(mode, WeightMode::Prefix) && coefficient_bits >= 2,
        "{label}"
    );
    let mut reference = Reference {
        witness: pad_compact_witness(&witness, live_lanes, lane_bits, coefficient_bits)
            .into_iter()
            .map(|w| F::from_i64(i64::from(w)))
            .collect(),
        linear,
        range,
    };
    let mut claim = reference.claim();
    assert_eq!(prover.input_claim(), claim, "input: {label}");
    for round in 0..num_vars {
        let expected = reference.round();
        let actual = prover.compute_round_univariate(round, claim);
        assert_eq!(actual, expected, "round {round}: {label}");
        assert_eq!(
            expected.evaluate(F::zero()) + expected.evaluate(F::one()),
            claim,
            "sum at round {round}: {label}"
        );
        let challenge = F::from_u64(101 + 13 * round as u64);
        claim = expected.evaluate(challenge);
        reference.bind(challenge);
        prover.ingest_challenge(round, challenge);
        assert_eq!(reference.claim(), claim, "fold at round {round}: {label}");
    }
    assert_eq!(prover.final_w_eval(), reference.witness[0], "w: {label}");
    assert_eq!(
        prover.expected_final_claim().unwrap(),
        reference.claim(),
        "final: {label}"
    );
}

#[test]
fn stage2_every_phase_matches_boolean_hypercube_reference() {
    for coefficient_bits in 1..=7 {
        for live_lanes in [19, 32] {
            for mode in [
                WeightMode::Prefix,
                WeightMode::Disabled,
                WeightMode::ReducedDense,
            ] {
                for sparse in [false, true] {
                    check_configuration(coefficient_bits, live_lanes, mode, sparse);
                }
            }
        }
    }
}
