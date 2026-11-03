use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=AKITA_CARRY_GRAPH_RS");
    if env::var_os("CARGO_FEATURE_CARRY_GRAPH").is_none() {
        return;
    }
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo output directory"));
    let source = match env::var_os("AKITA_CARRY_GRAPH_RS") {
        Some(path) => {
            let path = PathBuf::from(path);
            println!("cargo:rerun-if-changed={}", path.display());
            fs::read_to_string(path).expect("trusted generated carry graph source")
        }
        None => "pub(super) static GRAPHS: &[Graph<'static>] = &[];\n".into(),
    };
    fs::write(output.join("carry_graph.rs"), source).expect("write generated carry graph source");
}
