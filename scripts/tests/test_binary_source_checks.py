"""Independent arithmetic properties and negative controls for checkpoint checks."""

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
        self.assertGreaterEqual(self.report["unit_resultant_lower_bits"],
                                self.report["unit_resultant_upper_bits"])
        # Independent signs can collide after reduction modulo two.
        self.assertEqual(tuple(x % 2 for x in (1, 0, -1)),
                         tuple(x % 2 for x in (-1, 0, 1)))
        # Deterministically sign every support: distinct supports remain distinct.
        supports = list(product((0, 1), repeat=6))
        images = {tuple((bit * (-1 if i % 2 else 1)) % 2
                        for i, bit in enumerate(support)) for support in supports}
        self.assertEqual(len(images), len(supports))

    def test_projection_residuals_all_requested_geometries(self):
        for geometry in self.report["geometry"]:
            self.assertLess(2 * geometry["projection"], checks.P)
            self.assertTrue(geometry["projection_exact_lift"])
            self.assertLessEqual(geometry["projection_signed_bytes"], 12)
        self.assertEqual(Fraction(self.report["projection_error"]),
                         Fraction(1, checks.Q**7))

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
        for geometry in self.report["geometry"]:
            self.assertLess(geometry["parity_exact_lift_bound"], checks.P)
            self.assertLessEqual(geometry["carry_bound"], geometry["carry_positive_reach"])
            self.assertEqual(geometry["nine_digit_carry_fits"], geometry["n"] <= 23)
        self.assertLessEqual(self.report["value_honest_carry_bound"],
                             self.report["value_carry_interval"][1])
        self.assertLess(self.report["value_exact_lift_bound"], checks.P)

    def test_response_tail_cap_and_width512_negative_control(self):
        self.assertGreater(self.report["candidate_width512"]["cap"],
                           checks.balanced_interval(4)[1])

    def test_optional_supports_and_exact_minimum_weights(self):
        for arity, target in ((256, 152), (512, 128), (1024, 128)):
            weight = checks.minimum_support_weight(arity, target)
            smaller = sum(checks.comb(162, i) for i in range(weight))
            current = smaller + checks.comb(162, weight)
            self.assertLess(smaller, arity * (1 << target))
            self.assertGreaterEqual(current, arity * (1 << target))
        self.assertGreater(self.report["weight77_response_sizing"]["26"]["cap"],
                           checks.balanced_interval(4)[1])

    def test_exact_work_filter_debit(self):
        for degree in (1, 2, 647, 1294):
            bits = checks.required_work_bits(degree)
            self.assertLessEqual(degree * (1 << 128), checks.P * (1 << bits))
            self.assertGreater(degree * (1 << 128), checks.P * (1 << (bits - 1)))
        self.assertEqual(checks.required_work_bits(2, modulus=1 << 128), 1)

    def test_exact_committed_quotient_fallback(self):
        for fallback in self.report["quotient_fallback"].values():
            exact = Fraction(fallback["honest_carry_fraction"])
            self.assertLess(fallback["honest_carry"] - 1, exact)
            self.assertGreaterEqual(fallback["honest_carry"], exact)
            self.assertLessEqual(fallback["honest_carry"], fallback["carry_interval"][1])
            self.assertLess(fallback["exact_lift_bound"], checks.P)

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


class BinarySourcePrimeExtractionTests(unittest.TestCase):
    def test_linear_plane_extracts_before_coefficient_weights(self):
        degree, modulus = checks.SCALAR_DEGREE, checks.P
        rng = random.Random(278)
        sources = [[rng.randrange(-4, 4) for _ in range(degree)] for _ in range(4)]
        other_columns = [[rng.randrange(-4, 4) for _ in range(degree)]
                         for _ in range(4)]
        row_weights = [2, 3, 5, 7]
        omega = [rng.randrange(modulus) for _ in range(degree)]

        def plane(components):
            return [sum(e * v[t] for e, v in zip(row_weights, components)) % modulus
                    for t in range(degree)]

        expected = plane(sources)
        # Either deterministic sign of the singleton support must work.
        for sign in (-1, 1):
            s = [0, sign] + [0] * (degree - 2)
            # Z^(-1) = -Z^(d-1)-Z^(d/2-1) modulo Z^d+Z^(d/2)+1.
            inverse = [0] * degree
            inverse[degree - 1] = inverse[degree // 2 - 1] = -sign
            self.assertEqual(checks.ring_product(s, inverse, -1),
                             [1] + [0] * (degree - 1))
            response = [[x + y for x, y in zip(checks.ring_product(s, f, -1), other)]
                        for f, other in zip(sources, other_columns)]
            neighbor = other_columns  # c'=0; all other fold coordinates are fixed.
            difference = [[x - y for x, y in zip(z, z_prime)]
                          for z, z_prime in zip(response, neighbor)]
            left = [(x - y) % modulus for x, y in zip(plane(response), plane(neighbor))]
            right = [x % modulus for x in checks.ring_product(s, expected, -1)]
            self.assertEqual(left, right)
            extracted = [[x % modulus for x in checks.ring_product(inverse, a, -1)]
                         for a in difference]
            extracted_plane = [x % modulus for x in checks.ring_product(inverse, left, -1)]
            self.assertEqual(extracted_plane, plane(extracted))
            self.assertEqual(extracted_plane, expected)
            self.assertEqual(checks.dot(omega, extracted_plane) % modulus,
                             sum(e * checks.dot(omega, f)
                                 for e, f in zip(row_weights, sources)) % modulus)

    def test_constant_coefficient_functional_does_not_extract(self):
        degree = checks.SCALAR_DEGREE
        zero = [0] * degree
        one = [1] + [0] * (degree - 1)
        matrix_entry = list(range(checks.D))
        image = matrix_entry  # T=A*1, with all coefficients already below q.
        for sign in (-1, 1):
            c = [0, sign] + [0] * (degree - 2)
            c_prime = zero
            z_scalar = c
            z_prime_scalar = zero
            ep = zero
            z = [0] * checks.D
            z[4] = -sign  # epsilon*iota(Z) = -epsilon*Y^4.
            self.assertEqual([x % checks.Q for x in checks.ring_product(matrix_entry, z)],
                             [x % checks.Q for x in checks.ring_product(z, image)])
            self.assertEqual(checks.ring_product(matrix_entry, [0] * checks.D),
                             [0] * checks.D)
            # The invalid map is E-linear but not linear over E[Z]/Phi.
            bad_z = [z_scalar[0]] + [0] * (degree - 1)
            bad_z_prime = [z_prime_scalar[0]] + [0] * (degree - 1)
            self.assertEqual(bad_z, checks.ring_product(c, ep, -1))
            self.assertEqual(bad_z_prime, checks.ring_product(c_prime, ep, -1))
            inverse = [0] * degree
            inverse[degree - 1] = inverse[degree // 2 - 1] = -sign
            extracted = checks.ring_product(inverse, z_scalar, -1)
            self.assertEqual(extracted, one)
            bad_extracted = [extracted[0]] + [0] * (degree - 1)
            self.assertNotEqual(bad_extracted, ep)
            self.assertNotEqual(bad_z, checks.ring_product(c, bad_extracted, -1))
            # Keeping the component as a plane instead forces EP=1.
            self.assertEqual(z_scalar, checks.ring_product(c, one, -1))
            self.assertEqual(one[0], 1)  # Constant-coefficient omega is applied last.


class BinarySourceAdjointTests(unittest.TestCase):
    def test_small_exhaustive_and_full_degree_random(self):
        checks.check_adjoint()

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
        # A unit weight at the last high-degree reduction destination and a
        # constant operand make the full convolution's final coefficient nonzero.
        degree = checks.D
        rho = [0] * degree
        rho[degree // 2 - 2] = 1
        operand = [1] + [0] * (degree - 1)
        extended = [
            sum(rho[index] * coefficient for index, coefficient in
                checks.monomial_reduction(degree, 1, exponent))
            for exponent in range(2 * degree - 1)
        ]
        product = checks.convolution(extended, operand[::-1])
        self.assertEqual(len(product), 3 * degree - 2)
        self.assertNotEqual(product[-1], 0)


if __name__ == "__main__":
    unittest.main()
