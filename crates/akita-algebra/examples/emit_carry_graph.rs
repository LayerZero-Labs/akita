#[cfg(feature = "carry-graph")]
fn main() {
    let exponent: u32 = std::env::args().nth(1).expect("exponent").parse().expect("integer exponent");
    let source = akita_algebra::offset_eq::carry_graph::emit_fixed_source(exponent)
        .expect("supported geometry");
    print!("{source}");
}

#[cfg(not(feature = "carry-graph"))]
fn main() {}
