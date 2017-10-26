ndarray-linalg
===============
[![Crate](http://meritbadge.herokuapp.com/ndarray-linalg)](https://crates.io/crates/ndarray-linalg)
[![docs.rs](https://docs.rs/ndarray-linalg/badge.svg)](https://docs.rs/ndarray-linalg)
[![wercker status](https://app.wercker.com/status/f04aeba682ea6e79577e15bd946344a5/s/master "wercker status")](https://app.wercker.com/project/byKey/f04aeba682ea6e79577e15bd946344a5)

Linear algebra package for Rust.

Dependencies
-------------

- [bluss/rust-ndarray](https://github.com/bluss/rust-ndarray)
- [stainless-steel/lapack](https://github.com/stainless-steel/lapack)

and more (See Cargo.toml).

Feature flags
--------------

- OpenBLAS
  - `openblas-static`: use OpenBLAS with static link (default)
  - `openblas-shared`: use OpenBLAS with shared link
  - `openblas-system`: use system OpenBLAS (experimental)
- Netlib
  - `netlib-static`: use Netlib with static link (default)
  - `netlib-shared`: use Netlib with shared link
  - `netlib-system`: use system Netlib (experimental)
- Intel MKL
  - `intel-mkl`: use Intel MKL shared link (experimental)

Examples
---------
See [examples](https://github.com/termoshtt/ndarray-linalg/tree/master/examples) directory.

