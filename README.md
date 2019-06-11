ndarray-linalg
===============
[![Crate](http://meritbadge.herokuapp.com/ndarray-linalg)](https://crates.io/crates/ndarray-linalg)
[![docs.rs](https://docs.rs/ndarray-linalg/badge.svg)](https://docs.rs/ndarray-linalg)
[![Build Status](https://dev.azure.com/rust-ndarray/ndarray-linalg/_apis/build/status/rust-ndarray.ndarray-linalg?branchName=master)](https://dev.azure.com/rust-ndarray/ndarray-linalg/_build/latest?definitionId=1&branchName=master)

Linear algebra package for Rust with [ndarray](https://github.com/bluss/ndarray) based on external LAPACK implementations.

Examples
---------
See [examples](https://github.com/termoshtt/ndarray-linalg/tree/master/examples) directory.

**Note**: To run examples, you must specify which backend will be used (as described below).
For example, you can execute the [solve](examples/solve.rs) example with the OpenBLAS backend like this:

```sh
cargo run --example solve --features=openblas
```

and run all tests of ndarray-linalg with OpenBLAS

```sh
cargo test --features=openblas
```

BLAS/LAPACK Backend
-------------------

Three BLAS/LAPACK implementations are supported:

- [OpenBLAS](https://github.com/cmr/openblas-src)
  - needs `gfortran` (or other Fortran compiler)
- [Netlib](https://github.com/cmr/netlib-src)
  - needs `cmake` and `gfortran`
- [Intel MKL](https://github.com/termoshtt/rust-intel-mkl) (non-free license, see the linked page)

There are three features corresponding to the backend implementations (`openblas` / `netlib` / `intel-mkl`):

```toml
[dependencies]
ndarray = "0.12"
ndarray-linalg = { version = "0.10", features = ["openblas"] }
```

### For librarian
If you creating a library depending on this crate, we encourage you not to link any backend:

```toml
[dependencies]
ndarray = "0.12"
ndarray-linalg = "0.10"
```

### Link backend crate manually
For the sake of linking flexibility, you can provide LAPACKE implementation (as an `extern crate`) yourself.
You should link a LAPACKE implementation to a final crate (like binary executable or dylib) only, not to a Rust library.

```toml
[dependencies]
ndarray = "0.12"
ndarray-linalg = "0.10"
openblas-src = "0.7" # or another backend of your choice

```

You must add `extern crate` to your code in this case:

```rust
extern crate ndarray;
extern crate ndarray_linalg;
extern crate openblas_src; // or another backend of your choice
```
