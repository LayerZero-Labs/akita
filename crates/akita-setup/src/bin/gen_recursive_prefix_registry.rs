fn main() {
    if let Err(err) = run() {
        eprintln!("failed to generate recursive setup-prefix registry: {err}");
        std::process::exit(1);
    }
}

#[cfg(feature = "disk-persistence")]
fn run() -> Result<(), akita_field::AkitaError> {
    let targets = std::env::args().skip(1).collect::<Vec<_>>();
    let targets = if targets.is_empty() {
        vec!["all".to_string()]
    } else {
        targets
    };
    for target in targets {
        match target.as_str() {
            "all" => {
                for path in akita_setup::generate_recursive_profile_prefix_registries()? {
                    println!("wrote recursive setup-prefix registry: {}", path.display());
                }
            }
            "scalar" => {
                let path = akita_setup::generate_recursive_scalar_profile_prefix_registry()?;
                println!(
                    "wrote recursive scalar setup-prefix registry: {}",
                    path.display()
                );
            }
            "multi-group" => {
                let path = akita_setup::generate_recursive_example_prefix_registry()?;
                println!(
                    "wrote recursive multi-group setup-prefix registry: {}",
                    path.display()
                );
            }
            other => {
                return Err(akita_field::AkitaError::InvalidSetup(format!(
                    "unknown recursive prefix registry target {other:?}; expected scalar, multi-group, or all"
                )));
            }
        }
    }
    Ok(())
}

#[cfg(not(feature = "disk-persistence"))]
fn run() -> Result<(), akita_field::AkitaError> {
    Err(akita_field::AkitaError::InvalidSetup(
        "gen_recursive_prefix_registry requires the disk-persistence feature".to_string(),
    ))
}
