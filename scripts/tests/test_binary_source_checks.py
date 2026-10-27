"""Regression values and independent negative controls for checkpoint checks."""

from fractions import Fraction
import importlib.util
from itertools import product
from pathlib import Path
import random
import unittest


SPEC = importlib.util.spec_from_file_location(
    "binary_source_checks", Path(__file__).resolve().parents[1] / "binary_source_checks.py"
)
checks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checks)


class BinarySourceArithmeticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = checks.arithmetic_report()

    def test_exhaustive_multiplier_and_both_reduction_signs(self):
        self.assertEqual(self.report["source_kernel"], {
            "product_multiplier": 972, "monomial_multiplier": 2,
            "reduction_multiplier": 3,
        })
        self.assertEqual(self.report["scalar_kernel"], {
            "product_multiplier": 243, "monomial_multiplier": 2,
            "reduction_multiplier": 3,
        })
        # Compare sparse recursion against independent polynomial long division.
        for degree, sign in product((2, 4, 6), (-1, 1)):
            for exponent in range(3 * degree):
                sparse = [0] * degree
                for index, coefficient in checks.monomial_reduction(degree, sign, exponent):
                    sparse[index] = coefficient
                dense = checks.reduce_polynomial([0] * exponent + [1], degree, sign)
                self.assertEqual(sparse, dense)
        self.assertNotEqual(checks.monomial_reduction(4, 1, 4),
                            checks.monomial_reduction(4, -1, 4))

    def test_unit_conditions_and_support_entropy(self):
        self.assertTrue(self.report["q_prime"])
        self.assertEqual(self.report["q_transform_remainder"], 1)
        self.assertEqual(self.report["largest_centered_congruent_prime"], checks.Q)
        self.assertEqual(checks.multiplicative_order(2, 243), 162)
        self.assertEqual(self.report["prime_order_mod_243"], 6)
        self.assertEqual(self.report["unit_resultant_lower_bits"], 762)
        self.assertEqual(self.report["unit_resultant_upper_bits"], 729)
        self.assertGreaterEqual(self.report["unit_resultant_lower_bits"],
                                self.report["unit_resultant_upper_bits"])
        self.assertAlmostEqual(self.report["support_bits"], 136.30027830905894)
        self.assertAlmostEqual(self.report["fold_debit_bits"], 128.30027830905894)
        # Independent signs can collide after reduction modulo two.
        self.assertEqual(tuple(x % 2 for x in (1, 0, -1)),
                         tuple(x % 2 for x in (-1, 0, 1)))
        # Deterministically sign every support: distinct supports remain distinct.
        supports = list(product((0, 1), repeat=6))
        images = {tuple((bit * (-1 if i % 2 else 1)) % 2
                        for i, bit in enumerate(support)) for support in supports}
        self.assertEqual(len(images), len(supports))

    def test_projection_residuals_all_requested_geometries(self):
        expected_residuals = {
            22: 66394525036769280,
            23: 132788824280985600,
            24: 265577422769418240,
        }
        expected_rank3 = {
            22: 1839701006388918996628930560,
            23: 3679395756376625911509811200,
            24: 7358785256352039741271572480,
        }
        for geometry in self.report["geometry"]:
            n, rank = geometry["n"], geometry["rank"]
            self.assertEqual(geometry["residual"], expected_residuals[n])
            self.assertEqual(3 * geometry["projection"], rank * expected_rank3[n])
            self.assertLess(2 * geometry["projection"], checks.P)
            self.assertTrue(geometry["projection_exact_lift"])
            self.assertEqual(geometry["projection_signed_bytes"], 12)
        self.assertEqual(Fraction(self.report["projection_error"]),
                         Fraction(1, checks.Q**7))
        self.assertAlmostEqual(self.report["projection_error_bits"], 166.3536257275425)

    def test_uniform_divisibility_probability_small_exhaustive_analogue(self):
        # Exhaust every nonzero residual, and every pair of independent rows.
        q = 3
        vectors = list(product(range(q), repeat=2))
        matrices = list(product(vectors, repeat=2))
        for residual in vectors[1:]:
            accepted = sum(all(checks.dot(row, residual) % q == 0 for row in matrix)
                           for matrix in matrices)
            self.assertEqual(Fraction(accepted, len(matrices)), Fraction(1, q**2))
        # Without the range restriction, field equality does not lift to Z.
        self.assertEqual((0 - checks.P) % checks.P, 0)
        self.assertNotEqual(0, checks.P)
        self.assertNotEqual(checks.P % checks.Q, 0)

    def test_response_parity_and_value_bounds(self):
        self.assertEqual(checks.balanced_interval(4), (-2340, 1755))
        self.assertEqual(checks.balanced_interval(8), (-9586980, 7190235))
        for geometry in self.report["geometry"]:
            self.assertLess(geometry["parity_exact_lift_bound"], checks.P)
            self.assertLessEqual(geometry["carry_bound"], geometry["carry_positive_reach"])
            self.assertEqual(geometry["nine_digit_carry_fits"], geometry["n"] <= 23)
        rank3 = {row["n"]: row for row in self.report["geometry"] if row["rank"] == 3}
        self.assertEqual(rank3[22]["parity_exact_lift_bound"], 185383496)
        self.assertEqual(rank3[23]["parity_exact_lift_bound"], 217233992)
        self.assertEqual(rank3[24]["carry_bound"], 63771648)
        self.assertEqual(self.report["value_honest_carry_bound"], 62208)
        self.assertEqual(self.report["value_carry_interval"], [-149796, 112347])
        self.assertEqual(self.report["value_exact_lift_bound"], 797257)
        self.assertLess(self.report["value_exact_lift_bound"], checks.P)
        self.assertEqual(self.report["agreement_bound"], 4279)
        self.assertEqual(self.report["source_comparison_radius"], 1506960)

    def test_response_tail_cap_and_width512_negative_control(self):
        self.assertEqual(self.report["response_sizing"], {
            "22": {"tail_log": 17, "cap": 1266},
            "23": {"tail_log": 18, "cap": 1303},
            "24": {"tail_log": 18, "cap": 1303},
        })
        self.assertEqual(self.report["candidate_width512"], {"tail_log": 18, "cap": 1862})
        self.assertGreater(self.report["candidate_width512"]["cap"],
                           checks.balanced_interval(4)[1])

    def test_optional_supports_and_exact_minimum_weights(self):
        self.assertEqual(self.report["minimum_support_weights"], {
            "arity256_target152": 77,
            "arity512_target128": 47,
            "arity1024_target128": 48,
        })
        for arity, target in ((256, 152), (512, 128), (1024, 128)):
            weight = checks.minimum_support_weight(arity, target)
            smaller = sum(checks.comb(162, i) for i in range(weight))
            current = smaller + checks.comb(162, weight)
            self.assertLess(smaller, arity * (1 << target))
            self.assertGreaterEqual(current, arity * (1 << target))
        profiles = self.report["support_profiles"]
        self.assertAlmostEqual(profiles["64"]["support_bits"], 154.2529482043)
        self.assertAlmostEqual(profiles["77"]["support_bits"], 160.220269844)
        self.assertEqual(profiles["64"]["source_comparison_radius"], 2096640)
        self.assertEqual(profiles["77"]["source_comparison_radius"], 2522520)
        self.assertEqual(self.report["weight77_response_sizing"], {
            "22": {"tail_log": 17, "cap": 1638},
            "23": {"tail_log": 18, "cap": 1685},
            "26": {"tail_log": 20, "cap": 1776},
        })
        self.assertGreater(self.report["weight77_response_sizing"]["26"]["cap"], 1755)

    def test_exact_work_filter_debit(self):
        self.assertEqual(self.report["work_bits"], {"1": 1, "2": 2, "647": 10, "1294": 11})
        for degree in (1, 2, 647, 1294):
            bits = checks.required_work_bits(degree)
            self.assertLessEqual(degree * (1 << 128), checks.P * (1 << bits))
            self.assertGreater(degree * (1 << 128), checks.P * (1 << (bits - 1)))
        self.assertEqual(checks.required_work_bits(2, modulus=1 << 128), 1)

    def test_exact_committed_quotient_fallback(self):
        honest = {"22": 4658146490, "23": 9316281203, "24": 18632550629}
        malicious = {"22": 626101272562449060, "23": 692495571806665380,
                     "24": 825284170295098020}
        for n, fallback in self.report["quotient_fallback"].items():
            self.assertEqual(fallback["honest_carry"], honest[n])
            exact = Fraction(fallback["honest_carry_fraction"])
            self.assertLess(honest[n] - 1, exact)
            self.assertGreaterEqual(honest[n], exact)
            self.assertEqual(fallback["carry_interval"], [-39268272420, 29451204315])
            self.assertLessEqual(honest[n], fallback["carry_interval"][1])
            self.assertEqual(fallback["exact_lift_bound"], malicious[n])
            self.assertLess(malicious[n], checks.P)

    def test_balanced_offset_and_reduced_parity_identities(self):
        # Exhaust all four-digit integers and the full eight-digit image endpoints.
        lo, hi = checks.balanced_interval(4)
        for value in list(range(lo, hi + 1)) + list(checks.balanced_interval(8)):
            depth = 4 if lo <= value <= hi else 8
            offset = -checks.balanced_interval(depth)[0]
            packed = value + offset
            digits = [((packed >> (3 * i)) & 7) - 4 for i in range(depth)]
            self.assertEqual(sum(digit * 8**i for i, digit in enumerate(digits)), value)
            self.assertEqual(packed & 1, value % 2)
            self.assertEqual(digits[0] % 2, value % 2)
        # Honest U=1 and malicious allowed U=-3 have the same binary reading.
        # Exact carries absorb the difference; no U booleanity is needed.
        degree = 4
        coefficients = list(product((0, 1), repeat=degree))
        for e, u in product(coefficients, repeat=2):
            honest = checks.ring_product(e, u, -1)
            terminal = [x % 2 for x in honest]
            malicious = checks.ring_product(e, [x - 4 for x in u], -1)
            for actual in (honest, malicious):
                residual = [a - v for a, v in zip(actual, terminal)]
                self.assertTrue(all(value % 2 == 0 for value in residual))
                carry = [value // 2 for value in residual]
                self.assertEqual([v + 2 * k for v, k in zip(terminal, carry)], actual)


class BinarySourceAdjointTests(unittest.TestCase):
    def test_small_exhaustive_and_full_degree_random(self):
        self.assertEqual(checks.check_adjoint(), 1605)

    def test_rank_one_setup_claim(self):
        rng = random.Random(610)
        degree, rank, rows = 4, 2, 3
        coefficient_weights = [rng.randrange(17) for _ in range(degree)]
        row_weights = [rng.randrange(17) for _ in range(rows)]
        output_weights = [[rng.randrange(17) for _ in range(degree)] for _ in range(rank)]
        matrix = [[[rng.randrange(-4, 4) for _ in range(degree)]
                   for _ in range(rows)] for _ in range(rank)]
        left = 0
        right = 0
        for i in range(rank):
            adjoint = checks.adjoint_split_convolution(output_weights[i], coefficient_weights)
            for j in range(rows):
                vector = [row_weights[j] * x for x in coefficient_weights]
                left += checks.dot(output_weights[i], checks.ring_product(matrix[i][j], vector))
                right += checks.dot(matrix[i][j], [row_weights[j] * h for h in adjoint])
        self.assertEqual(left, right)

    def test_wrong_reduction_and_truncation_are_detected(self):
        rho, operand, vector = [1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 0, 1]
        target = checks.dot(rho, checks.ring_product(vector, operand))
        self.assertNotEqual(target, checks.dot(checks.adjoint_direct(rho, operand, -1), vector))
        # Omitting the extended Hankel row loses the high-exponent contribution.
        bad_product = checks.convolution(rho, operand[::-1])
        bad_adjoint = [bad_product[len(rho) - 1 + s] for s in range(len(rho))]
        self.assertNotEqual(target, checks.dot(bad_adjoint, vector))
        self.assertEqual(3 * checks.D - 2, 1942)
        self.assertEqual(2 * checks.D - 1, 1295)

    def test_literal_convolution_last_coefficient_is_nonzero(self):
        # rho[322]=1 gives g[1294]=-1. With v[0]=1, reverse(v)[647]=1,
        # so the last ordinary convolution coefficient is -1 at degree 1941.
        degree = checks.D
        last_extended = dict(checks.monomial_reduction(degree, 1, 2 * degree - 2))
        self.assertEqual(last_extended[322], -1)
        reverse_last = 1
        self.assertEqual(last_extended[322] * reverse_last, -1)
        self.assertEqual((2 * degree - 2) + (degree - 1), 1941)


if __name__ == "__main__":
    unittest.main()
