//! The residual opening as the Σ-protocol's statement
//! (`specs/whiteout-akita.md` §4.6, §4.8.1).
//!
//! At the cut the prover holds `(w_k, t̂_a, r_a)` with `A·G⁻¹(w_k) = G·t̂_a`
//! per block and `[B | B′](t̂_a, r_a) = u_a`. [`ResidualShape`] lays them out
//! as the residual segments of jolt-zk's private vector `s` and cuts the
//! opening rows from Akita's setup stream. Both roles build it from the
//! schedule's [`ResidualGeometry`], so the segment alphabets have one owner:
//! the schedule.

use std::ops::Range;
use std::sync::Arc;

use akita_error::AkitaError;
use akita_params::zk::{AdmittedZkSchedule, ResidualGeometry};
use jolt_algebra::{CanonicalEncoding, CyclotomicRing};
use jolt_crypto::lattice::{MatrixShape, MatrixView, PreparedMatrix, CHALLENGE_RING_DIM};
use jolt_crypto::CryptoError;
use jolt_poly::{Point, HIGH_TO_LOW};
use jolt_zk::whiteout::{
    BlockTarget, EvaluationClaims, ResidualBlock, ResidualGadget, Segment, SegmentMask,
    SegmentRole, SegmentShape,
};
use zeroize::Zeroizing;

use super::F;
use akita_prover::ResidualOpening;

/// The residual segments of `s` and the opening rows `N` of one
/// zero-knowledge schedule's cut.
///
/// The segments, in order:
///
/// - w_k, zero-padded to whole Akita blocks, one run per maximal stretch of
///   native elements that lie wholly inside or wholly outside the public
///   spans of [`ResidualGeometry::public_spans`] (public runs are
///   [`SegmentMask::Public`]; an element that straddles a span stays
///   Gaussian), at the cut level's open alphabet;
/// - t̂_a, one segment per Akita block, at its gadget basis;
/// - r_a, at the residual randomness alphabet.
///
/// Row block `b` of `N` is `A·w_b − G·t̂_b = 0`, and the last is
/// `[B | B′](t̂_a, r_a) = u_a`.
#[derive(Clone, Debug)]
pub struct ResidualShape {
    geometry: ResidualGeometry,
    segments: Vec<Segment>,
    /// Per Akita block: its witness segments and its inner-digit segment.
    block_segments: Vec<(Range<usize>, usize)>,
}

/// The residual's opening matrices `A` and `[B | B′]` over `R_D`
/// ([`ResidualShape::matrices`]).
pub struct ResidualMatrices<const D: usize> {
    a: Arc<PreparedMatrix<F, D>>,
    b: Arc<PreparedMatrix<F, D>>,
}

impl ResidualShape {
    /// The shape of an admitted schedule's cut.
    ///
    /// # Errors
    ///
    /// The errors of [`ResidualGeometry::new`];
    /// [`AkitaError::UnsupportedSchedule`] if the residual commitment's ring
    /// dimension is not a multiple of 64 with `d_A = d_B`, or unless the
    /// residual commits w_k in one inner digit plane whose alphabet holds
    /// w_k's (then `G⁻¹(w_k)` is w_k and the inner rows are `A·w_k`).
    pub fn new(admitted: AdmittedZkSchedule<'_>) -> Result<Self, AkitaError> {
        let geometry = ResidualGeometry::new(admitted, <F as CanonicalEncoding>::MODULUS_BITS)?;
        let profile = geometry.profile;
        let ring_dim = profile.inner.matrix.ring_dimension();
        if ring_dim % CHALLENGE_RING_DIM != 0 || profile.outer.matrix.ring_dimension() != ring_dim {
            return Err(AkitaError::UnsupportedSchedule(
                "the residual commitment needs d_A = d_B, a multiple of 64".into(),
            ));
        }
        if profile.inner.digits.num_digits != 1
            || geometry.witness_log_basis > profile.inner.digits.log_basis
        {
            return Err(AkitaError::UnsupportedSchedule(
                "the residual commitment must take w_k in one inner digit plane of its alphabet"
                    .into(),
            ));
        }
        let stride = ring_dim / CHALLENGE_RING_DIM;
        let segment = |role, elements: usize, log_basis, mask| Segment {
            role,
            len: elements * stride,
            native_dim: ring_dim,
            log_basis,
            shape: SegmentShape::Dense,
            mask,
        };
        let positions = profile.blocks.positions_per_block;
        let is_public = |element: usize| {
            let coefficients = element * ring_dim..(element + 1) * ring_dim;
            geometry
                .public_spans
                .iter()
                .any(|span| span.start <= coefficients.start && coefficients.end <= span.end)
        };
        let mut segments = Vec::new();
        let mut witness = Vec::with_capacity(profile.blocks.live_blocks);
        for block in 0..profile.blocks.live_blocks {
            let first = segments.len();
            let elements = block * positions..(block + 1) * positions;
            let mut run_start = elements.start;
            for element in elements.clone() {
                let next = element + 1;
                if next == elements.end || is_public(next) != is_public(run_start) {
                    let mask = if is_public(run_start) {
                        SegmentMask::Public
                    } else {
                        SegmentMask::Gaussian
                    };
                    segments.push(segment(
                        SegmentRole::Witness,
                        next - run_start,
                        geometry.witness_log_basis,
                        mask,
                    ));
                    run_start = next;
                }
            }
            witness.push(first..segments.len());
        }
        let inner_elements = profile.inner.matrix.output_rank() * profile.outer.digits.num_digits;
        let mut block_segments = Vec::with_capacity(witness.len());
        for runs in witness {
            block_segments.push((runs, segments.len()));
            segments.push(segment(
                SegmentRole::InnerDigits,
                inner_elements,
                profile.outer.digits.log_basis,
                SegmentMask::Gaussian,
            ));
        }
        segments.push(segment(
            SegmentRole::ResidualRandomness,
            profile.randomness_columns,
            geometry.randomness_log_basis,
            SegmentMask::Gaussian,
        ));
        Ok(Self {
            geometry,
            segments,
            block_segments,
        })
    }

    /// The schedule's residual geometry.
    pub fn geometry(&self) -> &ResidualGeometry {
        &self.geometry
    }

    /// The residual segments of `s`, in order.
    pub fn segments(&self) -> Vec<Segment> {
        self.segments.clone()
    }

    /// The residual commitment's ring dimension, the Σ-protocol's `D`.
    pub fn ring_dim(&self) -> usize {
        self.geometry.profile.inner.matrix.ring_dimension()
    }

    /// Coefficients of w_k's span in `s`: whole Akita blocks.
    fn witness_len(&self) -> usize {
        let blocks = self.geometry.profile.blocks;
        blocks.live_blocks * blocks.positions_per_block * self.ring_dim()
    }

    /// The evaluation claim `Witness(0) = w̃_k(point)`.
    ///
    /// # Errors
    ///
    /// [`AkitaError::InvalidInput`] unless `point` has exactly the variables
    /// of w_k's padded span, so that zero padding w_k keeps its multilinear
    /// extension.
    pub fn claims(&self, point: &Point<HIGH_TO_LOW, F>) -> Result<EvaluationClaims<F>, AkitaError> {
        let num_vars = self
            .witness_len()
            .max(self.ring_dim())
            .next_power_of_two()
            .trailing_zeros() as usize;
        if point.as_slice().len() != num_vars {
            return Err(AkitaError::InvalidInput(format!(
                "the residual point has {} variables, w_k's padded span {num_vars}",
                point.as_slice().len()
            )));
        }
        Ok(EvaluationClaims {
            points: vec![point.as_slice().to_vec()],
        })
    }

    /// The residual part of `s`: w_k zero-padded to whole blocks, then t̂_a,
    /// then r_a. Wiped when dropped.
    ///
    /// # Errors
    ///
    /// [`AkitaError::InvalidInput`] if a part's length differs from the
    /// shape's.
    pub fn residual(
        &self,
        opening: &ResidualOpening<F, F>,
    ) -> Result<Zeroizing<Vec<i64>>, AkitaError> {
        let profile = self.geometry.profile;
        let d = self.ring_dim();
        let t_hat_len = profile.blocks.live_blocks
            * profile.inner.matrix.output_rank()
            * profile.outer.digits.num_digits
            * d;
        if opening.w_k.len() > self.witness_len()
            || opening.t_hat_a.len() != t_hat_len
            || opening.r_a.len() != profile.randomness_columns * d
        {
            return Err(AkitaError::InvalidInput(
                "the residual opening differs from the schedule's residual shape".into(),
            ));
        }
        let mut s = Zeroizing::new(vec![
            0i64;
            self.witness_len() + t_hat_len + opening.r_a.len()
        ]);
        let (witness, rest) = s.split_at_mut(self.witness_len());
        let (inner, randomness) = rest.split_at_mut(t_hat_len);
        for (dst, &digit) in witness.iter_mut().zip(opening.w_k.iter()) {
            *dst = i64::from(digit);
        }
        for (dst, &digit) in inner.iter_mut().zip(opening.t_hat_a.iter()) {
            *dst = i64::from(digit);
        }
        for (dst, &digit) in randomness.iter_mut().zip(opening.r_a.iter()) {
            *dst = i64::from(digit);
        }
        Ok(s)
    }

    /// The shapes of `A` and `[B | B′]` over the residual ring.
    fn matrix_shapes(&self) -> (MatrixShape, MatrixShape) {
        let ring_dim = self.ring_dim();
        let profile = self.geometry.profile;
        let rows = profile.inner.matrix.output_rank();
        let inner_rings = profile.blocks.live_blocks * rows * profile.outer.digits.num_digits;
        (
            MatrixShape {
                rows,
                width: profile.blocks.positions_per_block,
                ring_dim,
            },
            MatrixShape {
                rows: profile.outer.matrix.output_rank(),
                width: inner_rings + profile.randomness_columns,
                ring_dim,
            },
        )
    }

    /// How many field elements of Akita's setup stream [`Self::matrices`]
    /// reads: both matrices are cut from its start.
    ///
    /// # Errors
    ///
    /// [`AkitaError::InvalidSetup`] if a matrix's size overflows `usize`.
    pub fn stream_len(&self) -> Result<usize, AkitaError> {
        let (a, b) = self.matrix_shapes();
        let len = |shape: MatrixShape| {
            shape
                .num_field_elements()
                .map_err(|error| AkitaError::InvalidSetup(error.to_string()))
        };
        Ok(len(a)?.max(len(b)?))
    }

    /// The opening matrices `A` and `[B | B′]`, cut from Akita's setup
    /// `stream`. Public: a function of the schedule and the setup alone, so
    /// a verifier builds them once and passes them to every
    /// [`Self::blocks`].
    ///
    /// # Errors
    ///
    /// [`AkitaError::InvalidInput`] if `D` is not the residual ring
    /// dimension, and the matrix errors of jolt-crypto.
    pub fn matrices<const D: usize>(
        &self,
        stream: &[F],
    ) -> Result<ResidualMatrices<D>, AkitaError> {
        if D != self.ring_dim() {
            return Err(AkitaError::InvalidInput(
                "the residual matrices' ring dimension differs from the schedule's".into(),
            ));
        }
        let crypto = |error: CryptoError| AkitaError::InvalidSetup(error.to_string());
        let prepare = |shape| -> Result<Arc<PreparedMatrix<F, D>>, AkitaError> {
            Ok(Arc::new(
                PreparedMatrix::new(&MatrixView::new(shape, stream).map_err(crypto)?)
                    .map_err(crypto)?,
            ))
        };
        let (a, b) = self.matrix_shapes();
        Ok(ResidualMatrices {
            a: prepare(a)?,
            b: prepare(b)?,
        })
    }

    /// The opening rows `N` over `matrices` ([`Self::matrices`]). `A` is the
    /// residual inner matrix, which takes w_k itself ([`Self::new`] checks
    /// one inner digit plane); `[B | B′]` is the outer matrix over
    /// `(t̂_a, r_a)`, and its target is u_a compressed: `payload` is `cm(u_a)`
    /// as the masked prefix sent it, and `image` is u_a, which only the
    /// prover holds (A47).
    ///
    /// # Errors
    ///
    /// [`AkitaError::InvalidInput`] if `D` is not the residual ring
    /// dimension, the matrices are not this shape's, or u_a has the wrong
    /// length.
    pub fn blocks<const D: usize>(
        &self,
        matrices: &ResidualMatrices<D>,
        payload: &[F],
        image: Option<&[F]>,
    ) -> Result<Vec<ResidualBlock<F, D>>, AkitaError> {
        let profile = self.geometry.profile;
        let rows = profile.inner.matrix.output_rank();
        let image_len = self.geometry.commitment_len()?;
        if D != self.ring_dim() || image.is_some_and(|u_a| u_a.len() != image_len) {
            return Err(AkitaError::InvalidInput(
                "the residual commitment differs from the schedule's residual shape".into(),
            ));
        }
        if (matrices.a.shape(), matrices.b.shape()) != self.matrix_shapes() {
            return Err(AkitaError::InvalidInput(
                "the residual matrices are another residual shape's".into(),
            ));
        }
        let (a, b) = (&matrices.a, &matrices.b);
        let mut blocks = self
            .block_segments
            .iter()
            .map(|(witness, inner)| ResidualBlock {
                matrix: Arc::clone(a),
                input: witness.clone(),
                gadget: Some(ResidualGadget {
                    segment: *inner,
                    log_basis: profile.outer.digits.log_basis,
                    levels: profile.outer.digits.num_digits,
                }),
                target: BlockTarget::Public(vec![CyclotomicRing::zero(); rows]),
            })
            .collect::<Vec<_>>();
        let first_inner = self.block_segments.first().map_or(0, |(_, inner)| *inner);
        let image = image
            .map(|u_a| {
                u_a.chunks_exact(D)
                    .map(|ring| {
                        let coefficients: [F; D] = ring.try_into().map_err(|_| {
                            AkitaError::Internal("u_a ring has D coefficients".into())
                        })?;
                        Ok(CyclotomicRing::from_coefficients(coefficients))
                    })
                    .collect::<Result<Vec<_>, AkitaError>>()
            })
            .transpose()?;
        blocks.push(ResidualBlock {
            matrix: Arc::clone(b),
            input: first_inner..self.segments.len(),
            gadget: None,
            target: BlockTarget::Compressed {
                payload: payload.to_vec(),
                image,
            },
        });
        Ok(blocks)
    }
}
