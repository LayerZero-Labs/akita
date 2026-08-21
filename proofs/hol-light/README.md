# Fp128 HOL Light proofs

These proofs check the standalone AArch64 addition and subtraction objects for
`Prime128OffsetA7F7`. Each proof contains the expected instruction words and
loads them with `define_assert_from_elf`. The proof fails if the object bytes
change.

The final theorems cover the callable functions, including the `ret`
instruction and the AArch64 procedure call convention. They assume canonical
field inputs and prove the canonical result modulo
`0xffffffffffffffffffffffff00005809`.

## Requirements

You need an AArch64 host, an OCaml environment that can build HOL Light proofs,
and local checkouts of HOL Light and `s2n-bignum`.

## Build fresh objects

Use a fresh Cargo target directory so the selected objects cannot come from an
older build.

```sh
AKITA_PROOF_TARGET="$(mktemp -d)"
CARGO_TARGET_DIR="$AKITA_PROOF_TARGET" \
  cargo build -p akita-field --release --features fp128-asm-experiment

AKITA_ADD_OBJECT="$(find "$AKITA_PROOF_TARGET/release/build" \
  -path '*/out/fp128_add.o' -print -quit)"
AKITA_SUB_OBJECT="$(find "$AKITA_PROOF_TARGET/release/build" \
  -path '*/out/fp128_sub.o' -print -quit)"
```

## Build and run the proofs

Set these paths for your local checkouts, then run the `s2n-bignum` proof
builder from its `arm` directory.

```sh
AKITA_DIR="$(pwd)"
HOL_LIGHT_DIR=/path/to/hol-light
S2N_BIGNUM_DIR=/path/to/s2n-bignum

cd "$S2N_BIGNUM_DIR/arm"
opam exec -- ../tools/build-proof.sh \
  "$AKITA_DIR/proofs/hol-light/fp128_add_correct.ml" \
  "$HOL_LIGHT_DIR/hol.sh" \
  "$AKITA_DIR/target/fp128_add_correct.native"
opam exec -- ../tools/build-proof.sh \
  "$AKITA_DIR/proofs/hol-light/fp128_sub_correct.ml" \
  "$HOL_LIGHT_DIR/hol.sh" \
  "$AKITA_DIR/target/fp128_sub_correct.native"

cd "$AKITA_DIR"
AKITA_FP128_ADD_OBJECT="$AKITA_ADD_OBJECT" \
  ./target/fp128_add_correct.native
AKITA_FP128_SUB_OBJECT="$AKITA_SUB_OBJECT" \
  ./target/fp128_sub_correct.native
```

Multiplication uses the same object linkage and benchmark path, but its HOL
Light correctness proof is not part of this experiment yet.
