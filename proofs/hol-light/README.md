# Fp128 HOL Light proofs

These proofs check the standalone AArch64 addition and subtraction objects for
`Prime128OffsetA7F7`. Each proof contains the expected instruction words and
loads them with `define_assert_from_elf`. The proof fails if the object bytes
change. Production A7F7 addition and subtraction include the same raw
instruction words.

Start with the Akita Book chapter
[Formal verification of arithmetic kernels](../../book/src/how/formal-verification.md)
if you are new to HOL Light or instruction semantics proofs. It explains the
register contract, the shape of an `ensures arm` theorem, the carry and borrow
arguments, and the syntax that differs from Lean.

The final theorems cover the callable functions, including the `ret`
instruction and the AArch64 procedure call convention. They assume canonical
field inputs and prove the canonical result modulo
`0xffffffffffffffffffffffff00005809`.

These are machine code theorems, not a proof of the whole Rust verifier. The
production checks confirm that the proved bodies occur in optimized public
operation witnesses and in the canonical verifier profile. The checks still
trust the Rust field invariant, the compiler, the formal AArch64 model, and the
physical processor. See [Exact claim and trust boundary](../../book/src/how/formal-verification.md#exact-claim-and-trust-boundary)
for the full boundary.

## Requirements

You need an AArch64 host, `llvm-objdump`, an OCaml environment that can build
HOL Light proofs, and local checkouts of HOL Light and `s2n-bignum`.

The CI workflow pins these revisions:

- HOL Light commit `433477862bb90b328a593e012e09390e99b2439b`
- `s2n-bignum` commit `ac31a43db30953037abd1b64b540e65cf31f4c67`

## Run every check

Set the paths to module built HOL Light and `s2n-bignum` checkouts. Then run the
same repository command used by CI.

```sh
HOL_LIGHT_DIR=/path/to/hol-light \
S2N_BIGNUM_DIR=/path/to/s2n-bignum \
  ./proofs/hol-light/check.sh
```

The script uses a new temporary Cargo target directory. It requires one fresh
addition object and one fresh subtraction object. It also builds optimized
production witnesses through the public field operations. The script checks
all addition and subtraction words. It builds the narrow release
`onehot_fp128` profile and requires both proved sequences inside verifier
symbols. It then rebuilds both native proof executables, runs both theorems,
and removes its temporary files when it exits.

Multiplication uses the same object linkage and benchmark path, but its HOL
