#!/usr/bin/env python3
"""Verify RS expansion correctness against Rust FFT output."""
import sys
import random

def main():
    p = 0xfffffffffffffffffffffffffffff6cd
    
    with open(sys.argv[1]) as f:
        lines = [int(line.strip()) for line in f if line.strip()]
    
    omega = lines[0]
    base_evals = lines[1:257]
    coeffs = lines[257:257+1470]
    extension = lines[257+1470:257+1470+1024]
    
    assert len(base_evals) == 256, f"Expected 256 base evals, got {len(base_evals)}"
    assert len(coeffs) == 1470, f"Expected 1470 coeffs, got {len(coeffs)}"
    assert len(extension) == 1024, f"Expected 1024 extension evals, got {len(extension)}"
    
    # Build padded_evals: base_evals + zeros
    padded_evals = base_evals + [0] * (1470 - 256)
    
    def eval_poly(coeffs, x, p):
        """Horner evaluation of polynomial at x mod p."""
        result = 0
        for c in reversed(coeffs):
            result = (result * x + c) % p
        return result
    
    random.seed(42)
    ok = True
    
    # Check base domain points
    test_base_indices = random.sample(range(256), 5)
    print(f"Checking {len(test_base_indices)} base domain points...")
    for i in test_base_indices:
        pt = pow(omega, i, p)
        got = eval_poly(coeffs, pt, p)
        expected = padded_evals[i]
        if got != expected:
            print(f"  FAIL: P(omega^{i}) = {got} != padded_evals[{i}] = {expected}")
            ok = False
        else:
            print(f"  PASS: P(omega^{i}) matches padded_evals[{i}]")
    
    # Check zero-padded region
    test_zero_indices = random.sample(range(256, 1470), 3)
    print(f"Checking {len(test_zero_indices)} zero-padded points...")
    for i in test_zero_indices:
        pt = pow(omega, i, p)
        got = eval_poly(coeffs, pt, p)
        expected = 0  # padded with zeros
        if got != expected:
            print(f"  FAIL: P(omega^{i}) = {got} != 0")
            ok = False
        else:
            print(f"  PASS: P(omega^{i}) == 0")
    
    # Check extension points
    test_ext_indices = random.sample(range(256, 1280), 5)
    print(f"Checking {len(test_ext_indices)} extension points...")
    for i in test_ext_indices:
        pt = pow(omega, i, p)
        got = eval_poly(coeffs, pt, p)
        expected = extension[i - 256]
        if got != expected:
            print(f"  FAIL: P(omega^{i}) = {got} != extension[{i-256}] = {expected}")
            ok = False
        else:
            print(f"  PASS: P(omega^{i}) matches extension[{i-256}]")
    
    if ok:
        print("\nAll checks PASSED.")
        sys.exit(0)
    else:
        print("\nSome checks FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    main()
