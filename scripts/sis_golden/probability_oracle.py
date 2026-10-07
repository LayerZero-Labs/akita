#!/usr/bin/env python3
"""Independent Decimal power-series oracle for SIS probability boundaries.

No Rust, libm, or lattice-estimator probability implementation is used. Run
with --write to refresh the fixtures, or without arguments to check them.
The 140-digit working precision retains over 100 digits after cancellation
in the erf series on the fixture domain x <= 8.
"""

import argparse
import csv
import io
import math
from decimal import Decimal as D, ROUND_CEILING, ROUND_FLOOR, localcontext
from pathlib import Path


def atan(x):
    term = x
    total = x
    n = 1
    while True:
        term *= -x * x
        updated = total + term / (2 * n + 1)
        if updated == total:
            return total
        total = updated
        n += 1


def erf(x, pi):
    term = x
    total = x
    n = 1
    while True:
        term *= -x * x / n
        updated = total + term / (2 * n + 1)
        if updated == total:
            return 2 * total / pi.sqrt()
        total = updated
        n += 1


def small_box(n, dimension, bound, beta, pi):
    log_two = D(2).ln()
    log_q = D(2**32 - 99).ln() / log_two
    beta = D(beta)
    step = ((beta / (2 * pi)).ln() - 1 + (pi * beta).ln() / beta)
    step /= (beta - 1) * log_two
    count = 1
    volume = n * log_q
    while step * count * (count + 1) / 2 <= volume and count < dimension:
        count += 1
    shift = (step * count * (count + 1) / 2 - volume) / count
    first = step * count - shift
    log_x = D(bound).ln() / log_two - D('0.5')
    log_x -= (D(4) / 3).ln() / (2 * log_two) + first
    log_x += D(dimension).ln() / (2 * log_two)
    mass = erf((log_x * log_two).exp(), pi)
    sieve_count = (D('0.2075') * beta * log_two).exp().to_integral_value(
        rounding=ROUND_FLOOR
    )
    log_p = dimension * mass.ln() + sieve_count.ln()
    p = min(log_p.exp(), D(1))
    repetitions = D(1) if p >= D('0.99') else (
        D('0.01').ln() / (1 - p).ln()
    ).to_integral_value(rounding=ROUND_CEILING)
    return log_x, p, repetitions, D('0.265') * beta + repetitions.ln() / log_two


def fixtures():
    with localcontext() as ctx:
        ctx.prec = 140
        pi = 16 * atan(D(1) / 5) - 4 * atan(D(1) / 239)
        log_two = D(2).ln()
        output = io.StringIO()
        writer = csv.writer(output, lineterminator='\n')
        writer.writerow(['log2_arg', 'log2_erf'])
        arguments = {-10000.0, -1000.0, -40.0}
        arguments.update(-20 + i / 8 for i in range(185))
        branch_points = [-20.0, 0.0, *[
            float(value.ln() / log_two)
            for value in [D('0.84375'), D('1.25'), 1 / D('0.35')]
        ]]
        for boundary in branch_points:
            arguments.update([boundary, math.nextafter(boundary, -math.inf),
                              math.nextafter(boundary, math.inf)])
        for arg in sorted(arguments):
            x = (D.from_float(arg) * log_two).exp()
            value = erf(x, pi).ln() / log_two
            writer.writerow([repr(arg), format(value, '.60g')])
        boundary_output = io.StringIO()
        boundaries = csv.writer(boundary_output, lineterminator='\n')
        boundaries.writerow(['n', 'm', 'bound', 'beta', 'log2_arg',
                             'probability', 'repetitions', 'quantum_cost'])
        # Both reported cells, neighbors on either side, and a pre-existing
        # active-dimension regression whose old expected value used rough erf.
        for n, dimension, bound, beta, stride in [
            (512, 462848, 1860, 483, 512),
            (96, 1145408, 1, 483, 32),
            (1024, 8192, 2**24 - 1, 343, 0),
        ]:
            for m in sorted({dimension - stride, dimension, dimension + stride}):
                values = small_box(n, m, bound, beta, pi)
                boundaries.writerow([n, m, bound, beta, *[
                    format(value, '.60g') for value in values
                ]])
        repetition_output = io.StringIO()
        repetitions = csv.writer(repetition_output, lineterminator='\n')
        repetitions.writerow(['log2_arg', 'dimension', 'log2_count', 'repetitions'])
        dimension = 462848
        log2_count = 100.2225
        log_count = D.from_float(log2_count) * log_two
        for attempts in [1, 2, 3]:
            target = 1 - (D('0.01').ln() / attempts).exp()
            target_mass = ((target.ln() - log_count) / dimension).exp()
            low, high = D(1), D(2)
            for _ in range(240):
                middle = (low + high) / 2
                if erf((middle * log_two).exp(), pi) < target_mass:
                    low = middle
                else:
                    high = middle
            center = float((low + high) / 2)
            for arg in [math.nextafter(center, -math.inf), center,
                        math.nextafter(center, math.inf)]:
                mass = erf((D.from_float(arg) * log_two).exp(), pi)
                p = (dimension * mass.ln() + log_count).exp()
                count = (D('0.01').ln() / (1 - p).ln()).to_integral_value(
                    rounding=ROUND_CEILING
                )
                repetitions.writerow([repr(arg), dimension, repr(log2_count), count])
        return {'probability_oracle.csv': output.getvalue(),
                'probability_boundaries.csv': boundary_output.getvalue(),
                'probability_repetition_boundaries.csv': repetition_output.getvalue()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    for name, content in fixtures().items():
        path = Path(__file__).with_name(name)
        if args.write:
            path.write_text(content)
        elif path.read_text() != content:
            raise SystemExit(f'{path}: oracle fixture differs; run with --write')
        print(f'{name}: verified independent Decimal oracle')


if __name__ == '__main__':
    main()
