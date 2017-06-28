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
//!  - [Scalar trait](types/trait.Field.html)
//!
//!  Usage
//!  ------
//!  Most functions in this crate is defined as [self-consuming trait technique][sct] like [serde]
//!  does.
//!
//!  For example, we can execute [eigh][eigh] using three types of interfaces:
//!
//!  ```rust,ignore
//!  let a = random((3, 3));
//!  let (eval, evec) = a.eigh(UPLO::Upper)?;
//!  let (eval, evec) = (&a).eigh(UPLO::Upper)?;
//!  let (eval, evec) = (&mut a).eigh(UPLO::Upper)?;
//!  ```
//!
//!  The first type `a.eigh()` consumes `a`, and the memory of `a` is used for `evec`.
//!  The second type `(&a).eigh()` consumes the reference (not `a` itself),
//!  and the memory for `evec` is newly allocated.
//!  The last one `(&mut a).eigh()` is similar to the first one;
//!  It borrows `a` mutably, and rewrite it to contains `evec`.
//!  In all cases, the array `eval` is newly allocated.
//!
//!  [sct]:https://github.com/serde-rs/serde/releases/tag/v0.9.0
//!  [serde]:https://github.com/serde-rs/serde

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
pub mod diagonal;

pub mod operator;
pub mod convert;
pub mod generate;
pub mod assert;
pub mod norm;
pub mod trace;

pub use assert::*;
pub use convert::*;
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
