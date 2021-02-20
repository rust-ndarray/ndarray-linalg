Linear Algebra eXtension (LAX)
===============================

ndarray-free safe Rust wrapper for LAPACK FFI for implementing ndarray-linalg crate.
This crate responsibles for

- Linking to LAPACK shared/static libraries
- Dispatching to LAPACK routines based on scalar types by using `Lapack` trait
