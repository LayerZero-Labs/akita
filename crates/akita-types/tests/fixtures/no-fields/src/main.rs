use akita_error::AkitaError;
use akita_types::dispatch_for_field;
use jolt_field::Prime64Offset59;

#[allow(dead_code)]
struct Degree<const D: usize>;

#[allow(dead_code)]
trait NeverImplemented {
    const D: usize;
}

fn main() -> Result<(), AkitaError> {
    let result: Result<usize, AkitaError> = dispatch_for_field!(
        ProtocolDispatchSlot::Ntt,
        Prime64Offset59,
        32,
        |D| Ok(<Degree<D> as NeverImplemented>::D)
    );
    let error = result.unwrap_err();
    assert!(error.to_string().contains("field-fp64"));
    Ok(())
}
