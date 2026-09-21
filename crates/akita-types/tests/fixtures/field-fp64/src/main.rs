use akita_error::AkitaError;
use akita_types::dispatch_for_field;
use jolt_field::Prime64Offset59;

struct Degree<const D: usize>;

trait Fp64CompressionDegree {
    const D: usize;
}

impl Fp64CompressionDegree for Degree<16> {
    const D: usize = 16;
}

impl Fp64CompressionDegree for Degree<32> {
    const D: usize = 32;
}

fn main() -> Result<(), AkitaError> {
    let dimension = dispatch_for_field!(
        ProtocolDispatchSlot::Compression,
        Prime64Offset59,
        32,
        |D| Ok(<Degree<D> as Fp64CompressionDegree>::D)
    )?;
    assert_eq!(dimension, 32);
    Ok(())
}
