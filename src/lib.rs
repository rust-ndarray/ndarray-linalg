//! The `ndarray-linalg` crate provides linear algebra functionalities for `ArrayBase`, the n-dimensional array data structure provided by [`ndarray`](https://github.com/rust-ndarray/ndarray).
//! `ndarray-linalg` leverages [LAPACK](http://www.netlib.org/lapack/)'s routines using the bindings provided by [stainless-steel/lapack](https://github.com/stainless-steel/lapack).
//!
//!  Linear algebra methods
//!  -----------------------
//!  - [QR decomposition](qr/trait.QR.html)
//!  - [singular value decomposition](svd/trait.SVD.html)
//!  - [solve linear problem](solve/index.html)
//!  - [solve linear problem for triangular matrix](triangular/trait.SolveTriangular.html)
//!  - [inverse matrix](solve/trait.Inverse.html)
//!  - [eigenvalue decomposition for Hermite matrix][eigh]
//!
//!  [eigh]:eigh/trait.Eigh.html
//!
//!  Utilities
//!  -----------
//!  - [assertions for array](index.html#macros)
//!  - [generator functions](generate/index.html)
//!  - [Scalar trait](types/trait.Scalar.html)

pub mod assert;
pub mod cholesky;
pub mod convert;
pub mod diagonal;
pub mod eigh;
pub mod error;
pub mod generate;
pub mod lapack;
pub mod layout;
pub mod norm;
pub mod operator;
pub mod opnorm;
pub mod qr;
pub mod solve;
pub mod solveh;
pub mod svd;
pub mod trace;
pub mod triangular;
pub mod types;

pub use assert::*;
pub use cholesky::*;
pub use convert::*;
pub use diagonal::*;
pub use eigh::*;
pub use generate::*;
pub use layout::*;
pub use norm::*;
pub use operator::*;
pub use opnorm::*;
pub use qr::*;
pub use solve::*;
pub use solveh::*;
pub use svd::*;
pub use trace::*;
pub use triangular::*;
pub use types::*;
