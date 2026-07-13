fn main() {
    if let Err(err) = run() {
        eprintln!("failed to generate recursive setup-prefix registry: {err}");
        std::process::exit(1);
    }
}

#[cfg(feature = "disk-persistence")]
fn run() -> Result<(), akita_field::AkitaError> {
    let path = akita_setup::generate_recursive_example_prefix_registry()?;
    println!("wrote recursive setup-prefix registry: {}", path.display());
    Ok(())
}

#[cfg(not(feature = "disk-persistence"))]
fn run() -> Result<(), akita_field::AkitaError> {
    Err(akita_field::AkitaError::InvalidSetup(
        "gen_recursive_prefix_registry requires the disk-persistence feature".to_string(),
    ))
}
