# Packed F162 round and fold kernels

Status: active
Book-chapter: book/src/foundations/field-arithmetic.md
Tracking: https://github.com/LayerZero-Labs/akita/issues/45

## Scope

Implementation and scoped acceptance checks are complete on the topic branch.
This record remains active until the implementation and Book chapter land;
then archive it under the spec lifecycle policy.

Add reusable computation storage and complete product-sumcheck arithmetic over
`BinaryField162`. This slice depends on the scalar arithmetic in PR #46. It does
not add transcript sampling, a sumcheck proof format, field switching, commitment
ownership, or a production binary PCS profile. Those contracts remain separate.

## Storage and arithmetic contract

`PackedBinary162` owns three equally sized arrays of 64-bit words. Entry `i`
uses the same low-degree-first polynomial coordinates as `BinaryField162`.
The last word has only 34 live bits. Construction accepts existing valid scalar
elements; private storage prevents callers from changing limb lengths or high
bits independently. This computation layout has no wire encoding. Scalar
encoding remains the existing canonical 21-byte representation.

`refill` replaces the contents and retains existing capacity when sufficient.
Folding retains capacity and updates storage in place; no full-table temporary
or per-round allocation is needed. Conversion and scalar extraction preserve
the exact element order.

For two equal-length tables `a` and `b`, a round pairs adjacent elements, so it
eliminates the least-significant table-index bit. A missing last odd-indexed
element is zero. With `da = a[2i] + a[2i+1]` and likewise `db`, the returned
coefficient array is:

```text
c0 = sum_i a[2i] b[2i]
c1 = sum_i (a[2i] db + da b[2i])
c2 = sum_i da db
g(X) = c0 + c1 X + c2 X^2
```

Addition is XOR. The caller supplies the current claim `h = g(0) + g(1)`.
The implementation computes only `c0` and `c2` and reconstructs `c1 = h + c2`,
saving one product per pair. The hint is trusted prover state: the kernel does
not authenticate it or rescan the tables to validate it. With an incorrect hint,
the returned polynomial's linear coefficient changes accordingly. Protocol
verification must authenticate the initial and terminal claims and round chain.
These coefficients require neither division by two nor interpolation at integer
points. `round_product` returns `None` for different
lengths or fewer than two entries. Empty and singleton tables have no next round.

`fold_in_place(r)` writes `a[2i] + r * da` and changes the length to its ceiling
half. Empty and singleton tables remain unchanged. Repeated odd-length folding
has the same result as padding the original table with zeros to a power of two.
Callers own the round count and public statement length; these kernels do not
authenticate padding or admit protocol parameters.

## Implementation constraints

Runtime feature selection occurs outside the element loops. Every hardware
entry point requires the features checked by its dispatcher. Loads and stores
stay inside the private arrays, including short tails and in-place overlap.
Portable arithmetic remains available and agrees with accelerated paths.
Round products may XOR-accumulate unreduced polynomials and reduce at the end:
XOR does not create an integer accumulation bound depending on the table length.

Reuse the existing canonical polynomial product and reduction operations where
applicable. Avoid a new backend trait hierarchy or a characteristic-two
implementation of the odd-characteristic `jolt_field::Field` contract.

## Acceptance

- [x] Complete round coefficients and fold sequences match an independent
  polynomial oracle, including zero/dense inputs and odd/vector-boundary tails.
- [x] Invalid lengths and terminal tables obey the stated API; refill after
  shrinking overwrites old contents and retains reusable storage.
- [x] Portable, ARM PMULL, and x86 PCLMUL/VPCLMUL kernels agree on supported CPUs.
- [x] Existing canonical scalar encoding and arithmetic tests remain valid.
- [x] Criterion compares complete rounds against the existing accelerated AoS
  dot-product API with reused gathering scratch. Report conversion, allocation,
  prepared message, and all-round costs separately; do not call a kernel-only
  measurement a protocol speedup.
- [x] Independent unsafe/correctness and maintainability review is complete.
