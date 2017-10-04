//!  Linear algebra package for [rust-ndarray](https://github.com/bluss/rust-ndarray) using LAPACK via [stainless-steel/lapack](https://github.com/stainless-steel/lapack)
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

extern crate lapack;
extern crate num_traits;
extern crate num_complex;
extern crate rand;
#[macro_use(s)]
extern crate ndarray;
#[macro_use]
extern crate procedurals;
#[macro_use]
extern crate derive_new;

pub mod assert;
pub mod cholesky;
pub mod convert;
pub mod diagonal;
pub mod eigh;
pub mod error;
pub mod generate;
pub mod lapack_traits;
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
