Linear Algebra eXtension (LAX)
===============================

[![crates.io](https://img.shields.io/badge/crates.io-lax-blue)](https://crates.io/crates/lax)
[![docs.rs](https://docs.rs/lax/badge.svg)](https://docs.rs/lax)

ndarray-free safe Rust wrapper for LAPACK FFI for implementing ndarray-linalg crate.
This crate responsibles for

- Linking to LAPACK shared/static libraries
- Dispatching to LAPACK routines based on scalar types by using `Lapack` trait
