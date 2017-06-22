ndarray-linalg
===============
[![Crate](http://meritbadge.herokuapp.com/ndarray-linalg)](https://crates.io/crates/ndarray-linalg)
[![docs.rs](https://docs.rs/ndarray-linalg/badge.svg)](https://docs.rs/ndarray-linalg)
[![wercker status](https://app.wercker.com/status/f04aeba682ea6e79577e15bd946344a5/s/master "wercker status")](https://app.wercker.com/project/byKey/f04aeba682ea6e79577e15bd946344a5)

Linear algebra package for [rust-ndarray](https://github.com/bluss/rust-ndarray) using LAPACK via [stainless-steel/lapack](https://github.com/stainless-steel/lapack)

Examples
---------
See [examples](https://github.com/termoshtt/ndarray-linalg/tree/master/examples) directory.

Versions
---------

- v0.5.0
    - **Breaking Change** Rewrite all algorithms to support complex numbers and general `ArrayBase`

- v0.4.1
    - ADD: assertion [#31](https://github.com/termoshtt/ndarray-linalg/pull/31)

- v0.4.0
    - MOD: use ndarray v0.9
