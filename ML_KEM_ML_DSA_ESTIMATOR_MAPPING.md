# ML-KEM / ML-DSA Estimator Mapping

This note answers a very concrete question:

> If we want to estimate the standardized NIST parameter sets for `ML-KEM` and
> `ML-DSA`, what exact tuples should we feed into an LWE / SIS estimator?

It also records what the local `lattice-estimator` in `../lattice-estimator`
currently reports for those tuples.

The important caveat up front is that there are three distinct layers:

1. the scheme parameters in the FIPS or CRYSTALS spec,
2. the intermediate structured hardness problem (`MLWE` or `MSIS`),
3. the final estimator tuple (`LWEParameters` or `SISParameters`) used by a
   concrete codebase.

The standardized schemes live at layer 1. A lattice estimator only sees layer 3.
So the entire exercise is about documenting the map from (1) to (3).

Main anchors:

- FIPS 203 parameter sets:
  `/tmp/nist_fips_203_ml_kem.pdf:1879-1923`
- FIPS 204 parameter sets:
  `/tmp/nist_fips_204_ml_dsa.pdf:890-929`
- Kyber PKE / KEM pseudocode and parameter rationale:
  `/tmp/kyber_round3_spec.pdf:332-490`, `/tmp/kyber_round3_spec.pdf:450-482`
- Dilithium overview, keygen/sign/verify rationale, and security discussion:
  `/tmp/dilithium_round3_spec.pdf:120-190`, `/tmp/dilithium_round3_spec.pdf:260-319`
- CRYSTALS security-estimate mappings:
  `/tmp/pqcrystals_Kyber.py:6-32`, `/tmp/pqcrystals_Dilithium.py:6-65`
- Local estimator tuple definitions:
  `../lattice-estimator/estimator/lwe_parameters.py:10-29`,
  `../lattice-estimator/estimator/sis_parameters.py:8-18`
- Local estimator built-in NIST Round 3 schemes:
  `../lattice-estimator/estimator/schemes.py:19-44`,
  `../lattice-estimator/estimator/schemes.py:131-182`
- Local estimator defaults:
  `../lattice-estimator/estimator/conf.py:10-17`

## 1. What The Local Estimator Wants

The local `lattice-estimator` exposes two relevant tuple types.

For LWE:

```python
LWEParameters(
    n,   # flat LWE secret dimension
    q,   # modulus
    Xs,  # secret distribution
    Xe,  # error distribution
    m,   # sample count
)
```

See `../lattice-estimator/estimator/lwe_parameters.py:10-29`.

For SIS:

```python
SISParameters(
    n,            # number of equations
    q,            # modulus
    length_bound, # bound on the short solution
    m,            # number of columns
    norm,         # 2 or oo
)
```

See `../lattice-estimator/estimator/sis_parameters.py:8-18`.

So the two main tasks are:

- flatten module dimensions to the estimator's `(n, m)`,
- and translate scheme-side bounded distributions or norm bounds into `Xs`,
  `Xe`, or `length_bound`.

## 2. ML-KEM

### 2.1 Rough algorithm

The PKE core of `ML-KEM` / Kyber is:

- sample a public matrix `A`,
- sample short secret `s` and short error `e`,
- publish `t = A s + e`,
- for encryption sample short ephemeral `r`, plus short errors `e1, e2`,
- compute
  - `u = A^T r + e1`,
  - `v = t^T r + e2 + encoded_message`,
- compress `(u, v)`,
- wrap the PKE with Fujisaki-Okamoto to obtain the KEM.

The exact role of `eta1` and `eta2` is visible in the Kyber pseudocode:

- key generation samples `s` and `e` from `B_{eta1}`,
- encryption samples `r` from `B_{eta1}`,
- encryption samples `e1, e2` from `B_{eta2}`.

See `/tmp/kyber_round3_spec.pdf:337-399`.

### 2.2 Standardized parameter sets

FIPS 203 gives:

| Parameter set | `n` | `q` | `k` | `eta1` | `eta2` | `du` | `dv` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ML-KEM-512` | `256` | `3329` | `2` | `3` | `2` | `10` | `4` |
| `ML-KEM-768` | `256` | `3329` | `3` | `2` | `2` | `10` | `4` |
| `ML-KEM-1024` | `256` | `3329` | `4` | `2` | `2` | `11` | `5` |

Anchor: `/tmp/nist_fips_203_ml_kem.pdf:1882-1886`.

### 2.3 Which parameters matter for lattice hardness?

The scheme parameters with direct lattice impact are:

- `n = 256`: ring degree.
- `q = 3329`: modulus.
- `k`: module rank. This is the main security-scaling knob.
- `eta1`: noise for `s`, `e`, and `r`.
- `eta2`: noise for `e1`, `e2`.
- `du`, `dv`: compression widths. These matter indirectly through deterministic
  rounding noise and failure behavior.

The rest of the KEM wrapper is crucial for full `IND-CCA2` security, but it is
not the thing a plain MLWE estimator directly measures.

### 2.4 CRYSTALS-to-estimator mapping

The reference CRYSTALS `security-estimates` script defines:

```python
def Kyber_to_MLWE(kps):
    ...
    return MLWEParameterSet(kps.n, kps.m, kps.m + 1, kps.ks, kps.q)
```

See `/tmp/pqcrystals_Kyber.py:21-32`.

This is the key modeling choice:

- `kps.n` is the ring degree `256`,
- `kps.m` is the module rank `k`,
- the estimator-facing `MLWEParameterSet` gets
  - `n = 256`,
  - `d = k`,
  - `m = k + 1`,
  - `k = eta1`,
  - `q = 3329`.

In words:

- the secret dimension is the module rank `k`,
- the sample count is taken to be `k + 1` ring samples,
- the noise parameter is taken to be the centered-binomial parameter `eta1`,
- after a sanity check that ciphertext rounding noise does not make the
  ciphertext-side MLWE instance weaker than the public-key one.

### 2.5 Flat local-estimator tuples

The local `lattice-estimator` does not expose a separate `MLWEParameterSet`.
Its `LWEParameters` are already flattened.

So the CRYSTALS-style tuples become:

| Parameter set | local `LWEParameters` input |
| --- | --- |
| `ML-KEM-512` | `n = 2*256`, `m = 3*256`, `q = 3329`, `Xs = CenteredBinomial(3)`, `Xe = CenteredBinomial(3)` |
| `ML-KEM-768` | `n = 3*256`, `m = 4*256`, `q = 3329`, `Xs = CenteredBinomial(2)`, `Xe = CenteredBinomial(2)` |
| `ML-KEM-1024` | `n = 4*256`, `m = 5*256`, `q = 3329`, `Xs = CenteredBinomial(2)`, `Xe = CenteredBinomial(2)` |

The local estimator also ships built-in `Kyber512/768/1024` tuples, but these are
slightly simpler:

- they use `m = k*256`,
- and the comment explicitly says they are "ignoring the compression."

See `../lattice-estimator/estimator/schemes.py:13-44`.

So for Kyber / ML-KEM there are really two reasonable local-estimator choices:

1. `local built-in`: the simplified local preset,
2. `CRYSTALS proxy`: the tuple that more closely matches the separate
   `pq-crystals/security-estimates` repo.

## 3. ML-DSA

### 3.1 Rough algorithm

The core of `ML-DSA` / Dilithium is:

- sample matrix `A`,
- sample short secrets `s1, s2`,
- form `t = A s1 + s2`,
- compress and publish high bits of `t`,
- to sign, sample a masking vector `y`,
- compute `w = A y`,
- keep only `w1 = HighBits(w, 2 gamma2)`,
- hash `(message, w1)` to a sparse challenge `c`,
- compute `z = y + c s1`,
- reject and restart if `z` or the low-bit checks violate the security /
  correctness bounds,
- provide small hints so the verifier can reconstruct the correct high bits.

See `/tmp/dilithium_round3_spec.pdf:120-190`.

### 3.2 Standardized parameter sets

FIPS 204 gives:

| Parameter set | `q` | `d` | `tau` | `lambda` | `gamma1` | `gamma2` | `(k,l)` | `eta` | `beta=tau*eta` | `omega` |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| `ML-DSA-44` | `8380417` | `13` | `39` | `128` | `2^17` | `(q-1)/88` | `(4,4)` | `2` | `78` | `80` |
| `ML-DSA-65` | `8380417` | `13` | `49` | `192` | `2^19` | `(q-1)/32` | `(6,5)` | `4` | `196` | `55` |
| `ML-DSA-87` | `8380417` | `13` | `60` | `256` | `2^19` | `(q-1)/32` | `(8,7)` | `2` | `120` | `75` |

Anchor: `/tmp/nist_fips_204_ml_dsa.pdf:890-909`.

### 3.3 Which parameters matter for lattice hardness?

The main lattice-relevant parameters are:

- `n = 256` implicitly from `R_q = Z_q[X]/(X^256 + 1)`,
- `q = 8380417`,
- `(k, l)` for matrix dimensions,
- `eta` for the secret coefficient range,
- `tau` for the challenge weight,
- `gamma1` for the masking range,
- `gamma2` for the low/high bit split,
- `d` for how many low-order bits are dropped from `t`.

The most important distinction from ML-KEM is:

- `ML-DSA` naturally gives both an `MLWE` problem and an `MSIS` problem.

The `MLWE` side comes from the public key relation:

```text
t = A s1 + s2
```

The `MSIS` side comes from forgery. In the Dilithium analysis this is the one
closer to the actual signature-security reduction.

### 3.4 CRYSTALS-to-estimator mapping

The CRYSTALS script defines the derived bounds:

```python
self.zeta       = max(gamma1, 2*gamma2 + 1 + 2**(pkdrop-1)*tau)
self.zeta_prime = max(2*gamma1, 4*gamma2 + 1)
```

See `/tmp/pqcrystals_Dilithium.py:15-18`.

Then it maps:

```python
def Dilithium_to_MSIS(dps, strong_uf=False):
    if strong_uf:
        return MSISParameterSet(dps.n, dps.k + dps.l + 1, dps.k, dps.zeta_prime, dps.q, norm="linf")
    else:
        return MSISParameterSet(dps.n, dps.k + dps.l + 1, dps.k, dps.zeta, dps.q, norm="linf")

def Dilithium_to_MLWE(dps):
    return MLWEParameterSet(dps.n, dps.l, dps.k, dps.eta, dps.q, distr="uniform")
```

See `/tmp/pqcrystals_Dilithium.py:57-65`.

This means:

- `MLWE` side:
  - ring degree `n = 256`,
  - secret dimension `d = l`,
  - sample count `m = k`,
  - uniform coefficient bound `eta`,
  - modulus `q = 8380417`.

- `MSIS` side:
  - ring degree `n = 256`,
  - number of equations `h = k`,
  - number of columns `w = k + l + 1`,
  - `l_infty` bound `B = zeta` for weak UF,
  - `l_infty` bound `B = zeta_prime` for strong UF.

For the standardized FIPS claim, the `strong UF` row is the more relevant one,
because FIPS 204 presents ML-DSA as `SUF-CMA`; see
`/tmp/nist_fips_204_ml_dsa.pdf:630-652`.

### 3.5 Flat local-estimator tuples

For the local `lattice-estimator`, the corresponding flat tuples are:

#### MLWE side

| Parameter set | local `LWEParameters` input |
| --- | --- |
| `ML-DSA-44` | `n = 4*256`, `m = 4*256`, `q = 8380417`, `Xs = Uniform(-2, 2)`, `Xe = Uniform(-2, 2)` |
| `ML-DSA-65` | `n = 5*256`, `m = 6*256`, `q = 8380417`, `Xs = Uniform(-4, 4)`, `Xe = Uniform(-4, 4)` |
| `ML-DSA-87` | `n = 7*256`, `m = 8*256`, `q = 8380417`, `Xs = Uniform(-2, 2)`, `Xe = Uniform(-2, 2)` |

#### MSIS side

The local estimator already ships these as built-ins:

| Parameter set | weak UF built-in | strong UF built-in |
| --- | --- | --- |
| `ML-DSA-44` | `Dilithium2_MSIS_WkUnf` | `Dilithium2_MSIS_StrUnf` |
| `ML-DSA-65` | `Dilithium3_MSIS_WkUnf` | `Dilithium3_MSIS_StrUnf` |
| `ML-DSA-87` | `Dilithium5_MSIS_WkUnf` | `Dilithium5_MSIS_StrUnf` |

See `../lattice-estimator/estimator/schemes.py:131-182`.

## 4. What The Local Estimator Currently Gives

All numbers below were computed with the local estimator's current defaults:

- reduction cost model `MATZOV`,
- reduction shape model `gsa`,
- `estimate.rough(...)` entrypoint.

See `../lattice-estimator/estimator/conf.py:10-17`.

### 4.1 ML-KEM results

#### Local built-ins

| Parameter set | tuple used | best algorithm | `log2(rop)` |
| --- | --- | --- | ---: |
| `Kyber512_local_builtin` | `n=512, m=512, q=3329, Xs=Xe=CenteredBinomial(3)` | `dual_hybrid` | `115.51` |
| `Kyber768_local_builtin` | `n=768, m=768, q=3329, Xs=Xe=CenteredBinomial(2)` | `dual_hybrid` | `174.32` |
| `Kyber1024_local_builtin` | `n=1024, m=1024, q=3329, Xs=Xe=CenteredBinomial(2)` | `dual_hybrid` | `241.76` |

#### CRYSTALS-style MLWE proxy

| Parameter set | tuple used | best algorithm | `log2(rop)` |
| --- | --- | --- | ---: |
| `ML-KEM-512_CRYSTALS_proxy` | `n=512, m=768, q=3329, Xs=Xe=CenteredBinomial(3)` | `dual_hybrid` | `115.51` |
| `ML-KEM-768_CRYSTALS_proxy` | `n=768, m=1024, q=3329, Xs=Xe=CenteredBinomial(2)` | `dual_hybrid` | `174.32` |
| `ML-KEM-1024_CRYSTALS_proxy` | `n=1024, m=1280, q=3329, Xs=Xe=CenteredBinomial(2)` | `dual_hybrid` | `241.76` |

Observation:

- Under the local estimator's current `rough` defaults, increasing the sample count
  from the local built-in `m = k*256` to the CRYSTALS-style `m = (k+1)*256`
  did not change the reported best `rop`.

That does **not** mean the tuples are identical as models. It means the current
best attack selected by the estimator is not improved by that extra sample count
in these specific cases.

### 4.2 ML-DSA results

#### MLWE side

| Parameter set | tuple used | best algorithm | `log2(rop)` |
| --- | --- | --- | ---: |
| `ML-DSA-44_MLWE` | `n=1024, m=1024, q=8380417, Xs=Xe=Uniform(-2,2)` | `dual_hybrid` | `122.64` |
| `ML-DSA-65_MLWE` | `n=1280, m=1536, q=8380417, Xs=Xe=Uniform(-4,4)` | `usvp` | `182.21` |
| `ML-DSA-87_MLWE` | `n=1792, m=2048, q=8380417, Xs=Xe=Uniform(-2,2)` | `dual_hybrid` | `245.10` |

#### MSIS side, weak UF

| Parameter set | tuple used | best algorithm | `log2(rop)` |
| --- | --- | --- | ---: |
| `ML-DSA-44_MSIS_weak` | `n=1024, m=2304, q=8380417, B=350209, norm=oo` | `lattice` | `123.52` |
| `ML-DSA-65_MSIS_weak` | `n=1536, m=3072, q=8380417, B=724481, norm=oo` | `lattice` | `186.30` |
| `ML-DSA-87_MSIS_weak` | `n=2048, m=4096, q=8380417, B=769537, norm=oo` | `lattice` | `265.43` |

#### MSIS side, strong UF

| Parameter set | tuple used | best algorithm | `log2(rop)` |
| --- | --- | --- | ---: |
| `ML-DSA-44_MSIS_strong` | `n=1024, m=2304, q=8380417, B=380929, norm=oo` | `lattice` | `121.76` |
| `ML-DSA-65_MSIS_strong` | `n=1536, m=3072, q=8380417, B=1048576, norm=oo` | `lattice` | `175.78` |
| `ML-DSA-87_MSIS_strong` | `n=2048, m=4096, q=8380417, B=1048576, norm=oo` | `lattice` | `253.46` |

### 4.3 The main takeaway

For `ML-DSA`, the two relevant local-estimator views are:

- `MLWE` for secret-key recovery / public-key hiding,
- `MSIS` for forgery.

Under the current local defaults, the smaller of those two for each standardized
set is:

| Parameter set | `MLWE` | `MSIS strong UF` | smaller one |
| --- | ---: | ---: | ---: |
| `ML-DSA-44` | `122.64` | `121.76` | `121.76` |
| `ML-DSA-65` | `182.21` | `175.78` | `175.78` |
| `ML-DSA-87` | `245.10` | `253.46` | `245.10` |

So, under the local estimator's current defaults:

- `ML-DSA-44`: the strong-UF `MSIS` side is slightly tighter,
- `ML-DSA-65`: the strong-UF `MSIS` side is clearly tighter,
- `ML-DSA-87`: the `MLWE` side is slightly tighter.

This is a useful sanity check:

- you really do want to look at **both** problems for Dilithium / ML-DSA,
- not just one of them.

## 5. Reproduction

The exact command used for the local-estimator sweep was a single `sage -python`
script run from `../lattice-estimator`, using:

- `LWE.estimate.rough(...)`,
- `SIS.estimate.rough(...)`,
- the built-in `schemes.Kyber*` and `schemes.Dilithium*_MSIS_*` where available,
- and manual `LWEParameters(...)` for the `ML-DSA` MLWE side.

If you want the same run again, the relevant objects are:

- local defaults: `../lattice-estimator/estimator/conf.py:10-17`
- built-in Kyber schemes: `../lattice-estimator/estimator/schemes.py:19-44`
- built-in Dilithium MSIS schemes: `../lattice-estimator/estimator/schemes.py:131-182`
- CRYSTALS-to-MLWE map for Kyber: `/tmp/pqcrystals_Kyber.py:21-32`
- CRYSTALS-to-MLWE/MSIS maps for Dilithium: `/tmp/pqcrystals_Dilithium.py:15-18`, `/tmp/pqcrystals_Dilithium.py:57-65`

## 6. Bottom Line

If I had to choose one estimator-facing tuple per standardized scheme family, I would use:

- `ML-KEM`: the CRYSTALS-style ciphertext-proxy `MLWE` tuple,
- `ML-DSA`: the strong-UF `MSIS` tuple, while still separately checking the
  `MLWE` tuple because it can be competitive or even smaller.

That is the cleanest way to stay faithful both to the standardized scheme
parameters and to the way the CRYSTALS teams themselves mapped those parameters
into concrete lattice-estimation inputs.
