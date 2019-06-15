FROM rust:1.35

RUN apt-get update \
 && apt-get install -y \
    cmake              \
    gfortran           \
    libssl-dev         \
    pkg-config         \
    zlib1g-dev         \
 && rm -rf /var/lib/apt/lists/*

RUN cargo install cargo-tarpaulin
