ndarray-linalg
===============
[![Crate](http://meritbadge.herokuapp.com/ndarray-linalg)](https://crates.io/crates/ndarray-linalg)
[![docs.rs](https://docs.rs/ndarray-linalg/badge.svg)](https://docs.rs/ndarray-linalg)
[![wercker status](https://app.wercker.com/status/f04aeba682ea6e79577e15bd946344a5/s/master "wercker status")](https://app.wercker.com/project/byKey/f04aeba682ea6e79577e15bd946344a5)

Linear algebra package for Rust.

Dependencies
-------------

- [bluss/rust-ndarray](https://github.com/bluss/rust-ndarray)
- [blas-lapack-rs/lapacke](https://github.com/blas-lapack-rs/lapacke)

and more (See Cargo.toml).

Choosing LAPACKE implementation
--------------------------------

For the sake of linking flexibility, you must provide LAPACKE implementation (as an `extern crate`) yourself.
Currently three LAPACKE implementations are supported and tested:

- [OpenBLAS](https://github.com/cmr/openblas-src)
- [Netlib](https://github.com/cmr/netlib-src)
- [Intel MKL](https://github.com/termoshtt/rust-intel-mkl) (non-free license, see the linked page)

You should link a LAPACKE implementation to a final crate (like binary executable or dylib) only, not to a Rust library.
Example:

`Cargo.toml`:
```toml
[depencdencies]
ndarray = "0.10"
ndarray-linalg = "0.8"
openblas-src = "0.5" # or another backend of your choice

```

`main.rs`:
```rust
extern crate ndarray;
extern crate ndarray_linalg;
extern crate openblas_src; // or another backend of your choice
```

Examples
---------
See [examples](https://github.com/termoshtt/ndarray-linalg/tree/master/examples) directory.

