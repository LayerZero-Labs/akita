# References

These public references provide background for the ideas used by Akita. They
are grouped by the part of the Book they help explain. The repository's code,
live specifications, generated tables, and tests define the behavior of the
current implementation.

## Lattice commitments and folding

- [LaBRADOR: Compact Proofs for R1CS from Module-SIS](https://eprint.iacr.org/2022/1341)
  develops compact lattice-based arguments and commitment techniques from
  Module-SIS.
- [Greyhound: Fast Polynomial Commitments from Lattices](https://eprint.iacr.org/2024/1293)
  develops a lattice-based polynomial commitment scheme with an emphasis on
  prover efficiency.
- [Hachi: Efficient Lattice-Based Multilinear Polynomial Commitments](https://eprint.iacr.org/2026/156)
  develops the lattice-folding lineage most directly related to Akita's ring
  relations.
- [Practical Product Proofs for Lattice Commitments](https://eprint.iacr.org/2020/517)
  gives background on proving multiplicative relations over lattice
  commitments.

## Multilinear polynomials and sum-check

- [Algebraic Methods for Interactive Proof Systems](https://doi.org/10.1145/146585.146605)
  introduces the sum-check protocol used to reduce a Boolean-cube sum to a
  single evaluation.
- [Some Improvements for the PIOP for ZeroCheck](https://eprint.iacr.org/2024/1210)
  explains equality-polynomial factoring techniques related to Akita's
  equality-factored sum-check.

## Extension fields and packed openings

- [Succinct Arguments over Towers of Binary Fields](https://eprint.iacr.org/2023/1784)
  develops packed multilinear techniques over extension fields and provides
  useful background for extension-opening reduction.

## Post-quantum lattice standards

- [NIST FIPS 203: Module-Lattice-Based Key-Encapsulation Mechanism](https://csrc.nist.gov/pubs/fips/203/final)
  standardizes a module-lattice key-encapsulation mechanism.
- [NIST FIPS 204: Module-Lattice-Based Digital Signature Standard](https://csrc.nist.gov/pubs/fips/204/final)
  standardizes a module-lattice digital signature scheme.

Akita is a polynomial commitment scheme, not an implementation of either NIST
standard. These standards are relevant because they show how structured
lattice assumptions are being deployed in production cryptography.

## Public implementations

The [Why lattices?](../introduction/why-lattices.md) chapter links to public
implementations of LaBRADOR, LaZer, RoKoko, and Jindo. They are useful points
of comparison for repository structure, portability, proof encoding, and
integration readiness.
