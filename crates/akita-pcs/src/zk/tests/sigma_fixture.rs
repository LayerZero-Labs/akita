//! New-field instance of Jolt's synthetic Whiteout fixture.
//!
//! Adapted from jolt-zk at 790c3fe260792e84107b18640b2093786fbc36e1. The upstream
//! test-support fixture fixes its field; this instance preserves its geometry,
//! relation and witnesses while using Akita's new protocol field. It invokes
//! the public Sigma prover/verifier, including the unchecked adversarial route.

use std::sync::Arc;

use jolt_algebra::{
    balanced_decompose_coefficients_pow2_i8_into, ring::cyclotomic::BalancedDecomposePow2Params,
    CyclotomicRing, Field, One, Prime128Offset6A23801F, Ring, Zero,
};
use jolt_crypto::lattice::{
    derive_public_matrix_prefix, sample_box_digits, MatrixShape, MatrixView, PreparedMatrix,
    PublicMatrixSeed, CHALLENGE_RING_DIM,
};
use jolt_poly::EqPolynomial;
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

use jolt_zk::masking::{
    AffineRow, MaskedOperand, PadIndex, PadLayout, ProductOperand, ProductPair, ProductRow,
    ResidualRelation, ResidualVar,
};
use jolt_zk::whiteout::{
    sample_zero_constant_term, BlockTarget, CompressedImage, EvaluationClaims, ImageCompression,
    LatticePadCommitment, LatticePadOpening, PadPlacement, ResidualBlock, ResidualGadget, Segment,
    SegmentMask, SegmentRole, SegmentShape, SigmaSetup, WhiteoutError,
};
const FIELD_LEVELS: usize = 32;

pub(super) type F = Prime128Offset6A23801F;
pub(super) const BASE_PADS: usize = 5;
pub(super) const PAD_RANK: usize = 2;
pub(super) const PAD_RANDOMNESS_COLUMNS: usize = 2;

pub(super) fn stream(len: usize, seed: u8) -> Vec<F> {
    derive_public_matrix_prefix(len, &PublicMatrixSeed::shake256_paged_v1([seed; 32])).unwrap()
}

pub(super) fn prepared<const D: usize>(
    rows: usize,
    width: usize,
    seed: u8,
) -> Arc<PreparedMatrix<F, D>> {
    let shape = MatrixShape {
        rows,
        width,
        ring_dim: D,
    };
    let data = stream(shape.num_field_elements().unwrap(), seed);
    Arc::new(PreparedMatrix::new(&MatrixView::new(shape, &data).unwrap()).unwrap())
}

pub(super) fn digits(rng: &mut impl RngCore, len: usize) -> Vec<i64> {
    (0..len)
        .map(|_| i64::from(rng.next_u32() % 16) - 8)
        .collect()
}

pub(super) fn ring_digits<const D: usize>(ring: &CyclotomicRing<F, D>) -> Vec<i64> {
    let params = BalancedDecomposePow2Params::<F>::new(FIELD_LEVELS, 4);
    let mut out = vec![0i8; FIELD_LEVELS * D];
    balanced_decompose_coefficients_pow2_i8_into(ring.coefficients(), &mut out, &params);
    out.into_iter().map(i64::from).collect()
}

/// Segment lengths, in challenge-ring elements; `scale` multiplies the
/// witness `w`.
pub(super) fn lengths<const D: usize>(scale: usize) -> [usize; 4] {
    let m = D / CHALLENGE_RING_DIM;
    [2 * m * scale, 4 * m * scale, FIELD_LEVELS * m, m]
}

pub(super) fn residual_segments<const D: usize>(scale: usize) -> Vec<Segment> {
    let [public, witness, inner, randomness] = lengths::<D>(scale);
    let segment = |role, len, mask| Segment {
        role,
        len,
        native_dim: D,
        log_basis: 4,
        shape: SegmentShape::Dense,
        mask,
    };
    vec![
        segment(SegmentRole::Witness, public, SegmentMask::Public),
        segment(SegmentRole::Witness, witness, SegmentMask::Gaussian),
        segment(SegmentRole::InnerDigits, inner, SegmentMask::Gaussian),
        segment(
            SegmentRole::ResidualRandomness,
            randomness,
            SegmentMask::Gaussian,
        ),
    ]
}

/// Inner `A` (rank 1 over the witness) and outer `B` (rank 2 over the inner
/// digits and the randomness).
pub(super) fn residual_matrices<const D: usize>(
    scale: usize,
) -> (Arc<PreparedMatrix<F, D>>, Arc<PreparedMatrix<F, D>>) {
    let [public, witness, inner, randomness] = lengths::<D>(scale);
    let m = D / CHALLENGE_RING_DIM;
    (
        prepared::<D>(1, (public + witness) / m, 11),
        prepared::<D>(2, (inner + randomness) / m, 12),
    )
}

pub(super) fn layout() -> PadLayout {
    PadLayout {
        operand_pads: vec![PadIndex(0), PadIndex(2)],
        ..PadLayout::new(BASE_PADS, 1).unwrap()
    }
}

pub(super) fn pad_commitment<const D: usize>() -> LatticePadCommitment<F, D> {
    let placement = PadPlacement::new(&layout()).unwrap();
    LatticePadCommitment::new(
        placement,
        &stream(1 << 18, 21),
        PAD_RANK,
        PAD_RANDOMNESS_COLUMNS,
    )
    .unwrap()
}

/// One synthetic instance: a residual witness `w` (a public and a Gaussian
/// part) under a two-level commitment, five masked messages and one masked
/// two-pair product row, over `R_D`.
pub(super) struct Instance<const D: usize> {
    pub(super) relation: ResidualRelation<F>,
    pub(super) claims: EvaluationClaims<F>,
    /// `w ‖ t̂ ‖ r_a`, then the pad digits.
    pub(super) s: Vec<i64>,
    pub(super) pads: LatticePadOpening,
    pub(super) u: Vec<CyclotomicRing<F, D>>,
    pub(super) t_p: Vec<CyclotomicRing<F, D>>,
    /// The multiple of the base witness length.
    pub(super) scale: usize,
}

impl<const D: usize> Instance<D> {
    /// Synthetic witness with the requested scale.
    pub(super) fn scaled(seed: u64, scale: usize) -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let c = CHALLENGE_RING_DIM;
        let [public, witness, _, randomness] = lengths::<D>(scale);
        let layout = layout();
        let pads: Vec<F> = (0..layout.num_pads).map(|_| F::random(&mut rng)).collect();
        let mut messages: Vec<F> = (0..BASE_PADS).map(|_| F::random(&mut rng)).collect();

        let w = digits(&mut rng, (public + witness) * c);
        let num_vars = w.len().next_power_of_two().trailing_zeros() as usize;
        let point: Vec<F> = (0..num_vars).map(|_| F::random(&mut rng)).collect();
        let eq = EqPolynomial::new(point.clone()).evaluations();
        messages[4] = w.iter().zip(&eq).map(|(&w, e)| F::from_i64(w) * *e).sum();
        let auxiliary = messages[0] * messages[1] + messages[2] * messages[3];
        let masked: Vec<F> = messages
            .iter()
            .chain(std::iter::once(&auxiliary))
            .zip(&pads)
            .map(|(m, p)| *m + *p)
            .collect();
        let slot = |index: usize| AffineRow::masked(masked[index], PadIndex(index));
        let mut relation = ResidualRelation::new(layout, 1);
        relation.affine.push(
            slot(2)
                .add_scaled(&slot(3), F::one())
                .add_scaled(&AffineRow::constant(-(messages[2] + messages[3])), F::one()),
        );
        relation
            .affine
            .push(AffineRow::variable(ResidualVar::Witness(0)).add_scaled(&slot(4), -F::one()));
        let pair = |x: usize, y: usize| ProductPair {
            x: ProductOperand::Masked(MaskedOperand {
                masked: masked[x],
                pad: PadIndex(x),
            }),
            y: slot(y),
        };
        relation.products.push(ProductRow {
            first: pair(0, 1),
            rest: vec![pair(2, 3)],
            z: slot(5),
        });

        let pads_commitment = pad_commitment::<D>();
        let g = sample_zero_constant_term::<F, CHALLENGE_RING_DIM, _>(&mut rng);
        let r_p = sample_box_digits(&mut rng, PAD_RANDOMNESS_COLUMNS * D, 4).unwrap();
        let (t_p, opening) = pads_commitment.commit(&pads, g, &r_p).unwrap();

        let (a, b) = residual_matrices::<D>(scale);
        let t = a.apply_centered(&w, 8).unwrap();
        let t_hat: Vec<i64> = t.iter().flat_map(ring_digits::<D>).collect();
        let r_a = digits(&mut rng, randomness * c);
        let outer: Vec<i64> = t_hat.iter().chain(&r_a).copied().collect();
        let u = b.apply_centered(&outer, 8).unwrap();
        let s = w.iter().chain(&t_hat).chain(&r_a).copied().collect();
        Self {
            relation,
            claims: EvaluationClaims {
                points: vec![point],
            },
            s,
            pads: opening,
            u,
            t_p,
            scale,
        }
    }

    /// The prover's targets `u` and `t_p`, compressed by `compression`.
    pub(super) fn targets(&self, compression: &dyn ImageCompression<F>) -> [BlockTarget<F, D>; 2] {
        [&self.u, &self.t_p].map(|image| {
            let flat: Vec<F> = image
                .iter()
                .flat_map(|ring| ring.coefficients().iter().copied())
                .collect();
            BlockTarget::Compressed {
                payload: compression.compress(&flat).unwrap().payload,
                image: Some(image.clone()),
            }
        })
    }

    /// The setup with targets `u` and `t_p`.
    pub(super) fn setup_with(&self, [u, t_p]: [BlockTarget<F, D>; 2]) -> SigmaSetup<F, D> {
        let (a, b) = residual_matrices::<D>(self.scale);
        let blocks = vec![
            ResidualBlock {
                matrix: a,
                input: 0..2,
                gadget: Some(ResidualGadget {
                    segment: 2,
                    log_basis: 4,
                    levels: FIELD_LEVELS,
                }),
                target: BlockTarget::Public(vec![CyclotomicRing::zero()]),
            },
            ResidualBlock {
                matrix: b,
                input: 2..4,
                gadget: None,
                target: u,
            },
        ];
        let pads = pad_commitment::<D>();
        let v_matrix = SigmaSetup::v_matrix_of(&blocks, &pads, 1, &stream(1 << 18, 31)).unwrap();
        SigmaSetup::new(
            residual_segments::<D>(self.scale),
            blocks,
            &pads,
            t_p,
            v_matrix,
        )
        .unwrap()
    }
}

/// The identity compression: the payload is the image and the witness its
/// radix-16 digit planes, as the statement held images before compression.
pub(super) struct RawImages;

impl ImageCompression<F> for RawImages {
    fn payload_len(&self, image_len: usize) -> Result<usize, WhiteoutError> {
        Ok(image_len)
    }

    fn witness_len(&self, image_len: usize) -> Result<usize, WhiteoutError> {
        Ok(image_len * FIELD_LEVELS)
    }

    fn compress(&self, image: &[F]) -> Result<CompressedImage<F>, WhiteoutError> {
        Ok(CompressedImage {
            payload: image.to_vec(),
            witness: {
                let mut digits = vec![0i8; image.len() * FIELD_LEVELS];
                balanced_decompose_coefficients_pow2_i8_into(
                    image,
                    &mut digits,
                    &BalancedDecomposePow2Params::<F>::new(FIELD_LEVELS, 4),
                );
                digits
            },
        })
    }

    fn reconstruct(&self, image_len: usize, witness: &[i8]) -> Result<Vec<F>, WhiteoutError> {
        if witness.len() != image_len * FIELD_LEVELS {
            return Err(WhiteoutError::ResponseRejected(
                "compression witness length",
            ));
        }
        Ok((0..image_len)
            .map(|i| {
                (0..FIELD_LEVELS).rev().fold(F::zero(), |acc, plane| {
                    acc * F::from_u64(16) + F::from_i64(i64::from(witness[plane * image_len + i]))
                })
            })
            .collect())
    }

    fn reconstruct_transpose(
        &self,
        image_len: usize,
        weights: &[F],
    ) -> Result<Vec<F>, WhiteoutError> {
        if weights.len() != image_len {
            return Err(WhiteoutError::InvalidRelation("image weights length"));
        }
        let mut out = vec![<F as Zero>::zero(); image_len * FIELD_LEVELS];
        let mut power = F::one();
        for plane in out.chunks_mut(image_len.max(1)) {
            for (value, weight) in plane.iter_mut().zip(weights) {
                *value = *weight * power;
            }
            power *= F::from_u64(16);
        }
        Ok(out)
    }

    fn check(&self, image_len: usize, payload: &[F], witness: &[i8]) -> Result<(), WhiteoutError> {
        if self.reconstruct(image_len, witness)? == payload {
            Ok(())
        } else {
            Err(WhiteoutError::ResponseRejected("compressed image"))
        }
    }
}
