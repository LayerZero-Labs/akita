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
- [Generating Hard Instances of Lattice Problems](https://doi.org/10.1145/237814.237838)
  by Ajtai introduces the short integer solution problem and the hash function
  behind what the Book calls an Ajtai commitment.
- [Worst-Case to Average-Case Reductions for Module Lattices](https://eprint.iacr.org/2012/090)
  by Langlois and Stehlé gives the hardness reduction for Module-SIS.
- [Short, Invertible Elements in Partially Splitting Cyclotomic Rings and Applications to Lattice-Based Zero-Knowledge Proofs](https://eprint.iacr.org/2017/523)
  by Lyubashevsky and Seiler is the source of the condition that the Book
  calls LS18: in a partially splitting ring, every short nonzero element is
  invertible.

## Multilinear polynomials and sum-check

- [Algebraic Methods for Interactive Proof Systems](https://doi.org/10.1145/146585.146605)
  introduces the sum-check protocol used to reduce a Boolean-cube sum to a
  single evaluation.
- [Some Improvements for the PIOP for ZeroCheck](https://eprint.iacr.org/2024/1210)
  explains equality-polynomial factoring techniques related to Akita's
  equality-factored sum-check. The `GruenSplitEq` type in `akita-algebra` is
  named after this split.

## Transcripts, encodings, and arithmetic

- [How To Prove Yourself: Practical Solutions to Identification and Signature Problems](https://doi.org/10.1007/3-540-47721-7_12)
  by Fiat and Shamir introduces the transform that replaces verifier
  randomness with a hash of the transcript.
- [Run-Length Encodings](https://doi.org/10.1109/TIT.1966.1053907)
  by Golomb introduces the code family behind the Golomb-Rice encoding of
  terminal response coefficients. Rice coding is the case with a power-of-two
  parameter.
- [The Residue Number System](https://doi.org/10.1109/TEC.1959.5219515)
  by Garner describes the mixed-radix reconstruction that the Book calls
  Garner reconstruction.

## Security estimation

- [Post-Quantum Key Exchange: A New Hope](https://eprint.iacr.org/2015/1092)
  by Alkim, Ducas, Pöppelmann, and Schwabe is the source of the lattice
  reduction cost model that the Book calls ADPS16.
- The CRYSTALS-Dilithium
  [round-3 specification](https://pq-crystals.org/dilithium/data/dilithium-specification-round3-20210208.pdf),
  Appendix C.3, and
  [NTRU Fatigue](https://eprint.iacr.org/2021/999) by Ducas and van Woerden
  give background for reduction-shape models of q-ary lattices. The Book's
  [security model](../how/security.md) explains the LGSA shape that the SIS
  estimator uses.
- [Finding Many Collisions via Reusable Quantum Walks](https://eprint.iacr.org/2022/676)
  and
  [Does Quantum Lattice Sieving Require Quantum RAM?](https://eprint.iacr.org/2024/1700)
  give background for the quantum cost discussion.

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
