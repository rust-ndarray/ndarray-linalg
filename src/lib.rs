//!  Linear algebra package for [rust-ndarray](https://github.com/bluss/rust-ndarray) using LAPACK via [stainless-steel/lapack](https://github.com/stainless-steel/lapack)

extern crate blas;
extern crate lapack;
extern crate num_traits;
extern crate num_complex;
extern crate rand;
#[macro_use(s)]
extern crate ndarray;
#[macro_use]
extern crate enum_error_derive;
#[macro_use]
extern crate derive_new;

#[macro_use]
pub mod types;
pub mod error;
pub mod layout;
pub mod lapack_traits;

pub mod cholesky;
pub mod eigh;
pub mod opnorm;
pub mod qr;
pub mod solve;
pub mod svd;
pub mod triangular;

pub mod generate;
pub mod assert;
pub mod norm;
pub mod trace;

pub use assert::*;
pub use generate::*;
pub use layout::*;
pub use types::*;

pub use cholesky::*;
pub use eigh::*;
pub use norm::*;
pub use opnorm::*;
pub use qr::*;
pub use solve::*;
pub use svd::*;
pub use trace::*;
pub use triangular::*;
