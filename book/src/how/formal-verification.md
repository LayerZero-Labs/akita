# Formal verification of arithmetic kernels

Akita uses handwritten assembly for some field arithmetic. Tests can compare
these kernels with a reference implementation on many inputs. Formal
verification answers a stronger question. It proves what the exact machine
instructions do for every input covered by the theorem.

This chapter explains the experimental HOL Light proofs for the AArch64
`Fp128` addition and subtraction kernels. It assumes no experience with HOL
Light. Readers who use Lean will find a short syntax guide near the end.

## Current scope

The formalization covers the standalone AArch64 objects for
`Prime128OffsetA7F7`.

| Operation | Object byte check | Arithmetic proof | Callable function proof |
| --- | --- | --- | --- |
| Addition | Complete | Complete | Complete |
| Subtraction | Complete | Complete | Complete |
| Multiplication | Not yet written | Not yet written | Not yet written |

The experiment does not replace the production arithmetic path. It creates a
stable object that HOL Light can inspect and a benchmark path for comparing
that object with the current inline assembly.

## What the proof connects

The proof has to connect three descriptions of one computation.

```text
field operation over integers modulo p
                 ^
                 | arithmetic theorem
                 |
formal AArch64 register and flag execution
                 ^
                 | instruction decoding
                 |
exact bytes loaded from the compiled object
```

The bottom connection prevents the proof from applying to an object with
different instructions. The middle connection describes each instruction in
the formal AArch64 model. The top connection states the field operation that
the resulting register values must represent.

The source path follows the same structure.

```text
crates/akita-field/asm/aarch64/fp128_*_body.inc
        |
        v
crates/akita-field/asm/aarch64/fp128_*.S
        |
        v
build.rs produces fp128_*.o
        |
        v
proofs/hol-light/fp128_*_correct.ml
```

The `.inc` file contains the arithmetic instructions. The `.S` file gives the
function a symbol and adds `ret`. The build script produces the object. The HOL
Light script loads that object and proves its behavior.

## Field and register representation

The field modulus is

$$
p = 2^{128} - C,
$$

where

$$
C = \mathtt{0xffffa7f7}
$$

and

$$
p = \mathtt{0xffffffffffffffffffffffff00005809}.
$$

The [prime fields chapter](../foundations/rings-and-fields.md#prime-fields)
explains arithmetic modulo a prime. The special form of this modulus lets the
kernel replace a subtraction of $p$ with an addition or subtraction of the
smaller value $C$.

Each field element uses two 64 bit words. If the low word is $a_0$ and the high
word is $a_1$, their integer value is

$$
a = \operatorname{val}(a_0) + 2^{64}\operatorname{val}(a_1).
$$

The callable assembly functions use these registers:

| Register | Value before the call |
| --- | --- |
| `x0` | Low word of $a$ |
| `x1` | High word of $a$ |
| `x2` | Low word of $b$ |
| `x3` | High word of $b$ |
| `x4` | $C$ |
| `x30` | Return address |

The function returns the low and high result words in `x0` and `x1`.

## The shape of a machine code theorem

The main statements use this form:

```ocaml
ensures arm
  precondition
  postcondition
  allowed_changes
```

The precondition describes the initial processor state. It states which bytes
are loaded at the program counter and which values are in the input registers.

The postcondition describes the final processor state. It states where the
program counter stops and what integer the output registers represent.

The final argument states which registers, flags, and other processor state may
change. This is called a frame condition. A proof with a precise frame
condition rules out an implementation that computes the right result while
silently changing an input register or memory that it promised to preserve.

The theorem quantifies over every input word. It does not enumerate test cases.
Its field arithmetic conclusion has the assumptions

$$
a < p \quad\text{and}\quad b < p.
$$

These assumptions say that both inputs are canonical field elements. The
machine execution is still modeled for other inputs, but the theorem does not
claim that those inputs produce a field result.

## Exact object binding

Each proof reads an object path from an environment variable. It then calls
`define_assert_from_elf` with an explicit list of 32 bit instruction words.

```ocaml
let akita_fp128_add_mc =
  define_assert_from_elf
    "akita_fp128_add_mc"
    akita_fp128_add_object
    [
      0xab020005;
      (* remaining instruction words *)
      0xd65f03c0
    ];;
```

The last word is the encoding of `ret`. HOL Light rejects the object if its
bytes differ from this list. This check prevents an old theorem from being
silently reused after someone changes the assembly.

`ARM_MK_EXEC_RULE` decodes the checked words. It produces the instruction rules
that the later tactics use to reason about registers and flags.

## Addition

For canonical inputs, the addition theorem states

$$
\operatorname{result} = (a+b) \bmod p.
$$

The kernel first adds the two 128 bit inputs. Let $l$ be the low 128 bits and
let $c_1$ be the carry bit. The proof establishes

$$
2^{128}c_1 + l = a+b.
$$

The kernel then adds $C$ to $l$. Let $t$ be the low 128 bits of this second sum
and let $c_2$ be its carry bit. The proof establishes

$$
2^{128}c_2 + t = l+C.
$$

Since $p=2^{128}-C$, the second carry tells us whether $l$ is at least $p$.
The kernel combines that fact with the carry from the original addition. It
selects $t$ exactly when the original sum needs reduction modulo $p$.

### Small addition example

Use an 8 bit word and the smaller modulus

$$
p = 2^8-5 = 251.
$$

Take $a=200$ and $b=100$. Their integer sum is 300. The first machine addition
keeps the low 8 bits, so it produces $l=44$ and a carry of one.

The correction candidate is

$$
t = l+5 = 49.
$$

The first carry says that reduction is required, so the kernel selects 49. This
is correct because

$$
300 \bmod 251 = 49.
$$

The real proof uses 128 bit words and covers every canonical pair. The small
example only shows why the two carry equations match modular reduction.

## Subtraction

For canonical inputs, the subtraction theorem states

$$
\operatorname{result} = (a+p-b) \bmod p.
$$

The expression includes $p$ because HOL Light natural number subtraction stops
at zero. Adding $p$ first keeps the expression nonnegative. It is the same field
operation as $a-b$ modulo $p$.

Let $d$ be the wrapped 128 bit result of subtracting $b$ from $a$. Let $r$ be
the borrow bit. The proof establishes

$$
2^{128}r + a = b+d.
$$

If there is no borrow, then $d=a-b$ and the kernel subtracts zero. If there is a
borrow, then

$$
d = 2^{128}+a-b.
$$

The kernel subtracts $C$ in that case. The corrected result is

$$
d-C = a+(2^{128}-C)-b = a+p-b.
$$

### Small subtraction example

Again use $p=251$ with an 8 bit word. Take $a=20$ and $b=30$. The machine
subtraction wraps and produces

$$
d = 256+20-30 = 246.
$$

The subtraction borrowed, so the kernel subtracts $C=5$. The result is 241.
This is the canonical field difference because

$$
(20+251-30) \bmod 251 = 241.
$$

## Instruction body theorem and callable theorem

Each proof exports two theorems.

The first theorem stops after the arithmetic instructions and before `ret`.
This form is easier to prove because the program counter advances in a simple
sequence.

The second theorem covers the complete callable function. It checks that `ret`
loads the program counter from `x30`. It also uses the AArch64 procedure call
rules to state which registers and flags a caller must expect the function to
change.

`ARM_ADD_RETURN_NOSTACK_TAC` derives the callable theorem from the instruction
body theorem. The function does not use stack memory, so no stack argument is
needed.

## How the proof script works

The tactics perform these jobs:

| Tactic | Job |
| --- | --- |
| `ARM_ACCSTEPS_TAC` | Execute the selected instructions symbolically |
| `DECARRY_RULE` | Turn carry and borrow facts into integer equations |
| `BOUNDER_TAC` | Prove that word values stay within their bit bounds |
| `ASM_CASES_TAC` | Prove each possible Boolean case |
| `ARITH_TAC` | Finish ordinary integer arithmetic |
| `ARM_ADD_RETURN_NOSTACK_TAC` | Add `ret` and the callable function contract |

Tactics are proof construction programs. They are not extra axioms. The HOL
Light kernel checks the theorem produced by every successful tactic script.

## A guide for Lean users

HOL Light proof files mix OCaml with quoted HOL terms. This is the main syntax
difference to notice when reading them.

| HOL Light | Similar Lean idea |
| --- | --- |
| ``let NAME = prove (`statement`, tactics);;`` | `theorem NAME : statement := by ...` |
| Text inside backticks | The proposition or expression being proved |
| Text outside backticks | OCaml that builds and applies tactics |
| `THEN` | Apply the next tactic to every goal produced by the previous tactic |
| `THENL [t1; t2]` | Give a separate tactic to each produced goal |
| `ABBREV_TAC` | Introduce a local name for an expression |
| `SUBGOAL_THEN` | State and prove an intermediate fact |
| `ASM_CASES_TAC` | Split on a proposition and keep each case as an assumption |
| `ASM_REWRITE_TAC` | Rewrite using the available assumptions |

A theorem name such as `AKITA_FP128_ADD_CORRECT` is an OCaml value whose type is
`thm`. The HOL Light kernel constructs this value through primitive inference
rules. Lean instead checks an elaborated proof term. In both systems, a tactic
cannot produce a theorem without passing through the trusted kernel.

## What the current proofs establish

The addition and subtraction proofs establish these facts:

- The selected objects contain the exact listed instructions.
- The instructions have the stated effect under the formal AArch64 model.
- Every canonical pair produces the correct field result.
- The callable functions return through `x30`.
- The functions respect the AArch64 register change policy.

## What remains outside the theorem

The current proofs do not establish these facts:

- The multiplication object is correct.
- Every Rust caller supplies canonical inputs and the expected value in `x4`.
- Every compiler emitted inline copy has the same bytes as the standalone
  object.
- The code has a particular execution time on physical hardware.
- The processor has no microarchitectural side channel.

The formal result relies on the HOL Light kernel and the AArch64 model used by
`s2n-bignum`. As with any instruction semantics proof, the result also relies on
that model matching the behavior of the physical processor.

## Source and review guide

Read the implementation and proofs in this order:

1. Read the shared [assembly instruction bodies](https://github.com/LayerZero-Labs/akita/tree/main/crates/akita-field/asm/aarch64).
2. Read the [build script](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-field/build.rs)
   that produces the objects.
3. Read the [experiment module](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-field/src/fp128_asm_experiment.rs)
   that exposes the inline and callable variants.
4. Read the [addition proof](https://github.com/LayerZero-Labs/akita/blob/main/proofs/hol-light/fp128_add_correct.ml).
5. Read the [subtraction proof](https://github.com/LayerZero-Labs/akita/blob/main/proofs/hol-light/fp128_sub_correct.ml).
6. Use the [proof instructions](https://github.com/LayerZero-Labs/akita/blob/main/proofs/hol-light/README.md)
   to build fresh objects and run the theorems.

When reviewing a future kernel, check the object byte binding first. Then check
the register precondition and frame condition. Finally, check that the integer
equations capture every carry, borrow, and reduction case needed by the field
operation.
