use akita_error::AkitaError;
use akita_types::dispatch_for_field;
use jolt_field::Prime64Offset59;

struct Degree<const D: usize>;

trait Fp64NttDegree {
    const D: usize;
}

macro_rules! impl_degrees {
    ($($dimension:literal),+ $(,)?) => {
        $(impl Fp64NttDegree for Degree<$dimension> {
            const D: usize = $dimension;
        })+
    };
}

impl_degrees!(32, 64, 128, 256, 512, 1024, 2048);

fn main() -> Result<(), AkitaError> {
    let dimension = dispatch_for_field!(
        ProtocolDispatchSlot::Ntt,
        Prime64Offset59,
        2048,
        |D| Ok(<Degree<D> as Fp64NttDegree>::D)
    )?;
    assert_eq!(dimension, 2048);
    Ok(())
}
