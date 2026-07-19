#[path = "digit_range/cases.rs"]
mod cases;
#[path = "digit_range/measurement.rs"]
mod measurement;

fn main() {
    if !measurement::run_requested() {
        eprintln!("digit_range_allocations requires --measure-case <case>");
        std::process::exit(2);
    }
}
