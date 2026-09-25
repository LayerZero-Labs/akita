# Akita fuzzing campaign

See the sections below (written after validation) for the full guide.

    cd fuzz
    cargo run --release -p akita-fuzz-runner -- prepare --out ../dist
    ../dist/akita-fuzz run --output /data/akita-fuzz
