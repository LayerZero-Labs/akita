use super::common;
use akita_config::proof_optimized::fp128;
use akita_cpu_backend::{CpuBackend, DensePoly, GroupContext};
use akita_params::BasisMode;
use akita_pcs::custom_source::{SignedBlockDecoder, SignedBlockPoly};
use akita_serialization::AkitaSerialize;
use jolt_field::{Ring, Zero};
use std::sync::Arc;

type Cfg = fp128::Dense;
type F = fp128::Field;
const NV: usize = 14;
const DOMAIN: &[u8] = b"akita/signed-block-poly-parity";

struct Decoder(Vec<i8>);

impl SignedBlockDecoder for Decoder {
    fn decode_i8(&self, start: usize, out: &mut [i8]) {
        out.copy_from_slice(&self.0[start..start + out.len()]);
    }
}

#[derive(Clone, Copy)]
enum Storage {
    I8,
    I16,
    Decoded,
}

fn roundtrip(storage: Storage, len: usize) {
    let pattern: &[i16] = match storage {
        Storage::I8 | Storage::Decoded => &[-128, -100, -65, -64, -1, 0, 1, 63, 64, 100, 127],
        Storage::I16 => &[-32768, -30000, -129, -128, -1, 0, 1, 127, 128, 30000, 32767],
    };
    let values: Vec<i16> = (0..len).map(|i| pattern[i % pattern.len()]).collect();
    let signed = match storage {
        Storage::I8 => SignedBlockPoly::i8(values.iter().map(|&v| v as i8).collect(), NV),
        Storage::I16 => SignedBlockPoly::i16(values.clone(), NV),
        Storage::Decoded => SignedBlockPoly::decoded_i8(
            Arc::new(Decoder(values.iter().map(|&v| v as i8).collect())),
            len,
            NV,
        ),
    }
    .unwrap();
    assert_eq!(signed.len(), len);
    assert_eq!(signed.num_vars(), NV);
    assert_eq!(signed.blocks().len(), len);
    let mut evals: Vec<F> = values.iter().map(|&v| F::from_i64(i64::from(v))).collect();
    evals.resize(1 << NV, F::zero());
    let dense = DensePoly::from_field_evals(NV, &evals).unwrap();
    let point = common::random_point(NV, 0x51b1_0c01);
    let openings = [common::dense_opening_lagrange(&evals, &point)];
    let scheme = common::load_workspace_scheme::<Cfg>(
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    )
    .unwrap();
    let setup = scheme.setup_prover(NV, 1).unwrap();
    let backend = CpuBackend::new(setup.expanded().clone()).unwrap();
    let signed_source = backend.import_source(vec![signed]).unwrap();
    let dense_source = backend.import_source(vec![dense]).unwrap();
    let signed_output = backend
        .commit(
            scheme.schedules(),
            &signed_source,
            GroupContext::scheduler_without_precommitted_groups(),
        )
        .unwrap();
    let dense_output = backend
        .commit(
            scheme.schedules(),
            &dense_source,
            GroupContext::scheduler_without_precommitted_groups(),
        )
        .unwrap();
    drop((signed_source, dense_source));
    let mut signed_bytes = Vec::new();
    let mut dense_bytes = Vec::new();
    signed_output
        .committed_group
        .serialize_compressed(&mut signed_bytes)
        .unwrap();
    dense_output
        .committed_group
        .serialize_compressed(&mut dense_bytes)
        .unwrap();
    assert_eq!(signed_bytes, dense_bytes, "commitment bytes");

    let verifier = common::self_derived_verifier(&scheme, &setup);
    let mut proof_bytes = Vec::new();
    for output in [signed_output, dense_output] {
        let opening = common::prove_input::<Cfg>(
            &point,
            &openings,
            &output.committed_group,
            output.private_handle,
            scheme.schedules(),
        );
        let proof = scheme
            .batched_prove(&setup, opening, &backend, DOMAIN, BasisMode::Lagrange)
            .unwrap();
        verifier
            .batched_verify(
                &proof,
                DOMAIN,
                common::verify_input::<Cfg>(
                    &point,
                    &openings,
                    &output.committed_group,
                    scheme.schedules(),
                ),
                BasisMode::Lagrange,
            )
            .unwrap();
        proof_bytes.push(proof);
    }
    assert_eq!(proof_bytes[0], proof_bytes[1], "proof bytes");
}

macro_rules! parity_test {
    ($name:ident, $storage:ident, $len:expr) => {
        #[test]
        fn $name() {
            common::init_rayon_pool();
            common::run_on_large_stack(|| roundtrip(Storage::$storage, $len));
        }
    };
}

parity_test!(signed_block_poly_i8_full, I8, 1 << NV);
parity_test!(signed_block_poly_i8_short, I8, 257);
parity_test!(signed_block_poly_i16_full, I16, 1 << NV);
parity_test!(signed_block_poly_i16_short, I16, 257);
parity_test!(signed_block_poly_decoded_full, Decoded, 1 << NV);
parity_test!(signed_block_poly_decoded_short, Decoded, 257);
