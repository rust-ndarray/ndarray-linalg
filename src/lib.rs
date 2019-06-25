//! The `ndarray-linalg` crate provides linear algebra functionalities for `ArrayBase`, the n-dimensional array data structure provided by [`ndarray`](https://github.com/rust-ndarray/ndarray).
//!
//! `ndarray-linalg` leverages [LAPACK](http://www.netlib.org/lapack/)'s routines using the bindings provided by [blas-lapack-rs/lapack](https://github.com/blas-lapack-rs/lapack).
//!
//! Linear algebra methods
//! -----------------------
//! - Decomposition methods:
//!     - [QR decomposition](qr/index.html)
//!     - [Cholesky/LU decomposition](cholesky/index.html)
//!     - [Eigenvalue decomposition for Hermite matrices](eigh/index.html)
//!     - [**S**ingular **V**alue **D**ecomposition](svd/index.html)
//! - Solution of linear systems:
//!    - [General matrices](solve/index.html)
//!    - [Triangular matrices](triangular/index.html)
//!    - [Hermitian/real symmetric matrices](solveh/index.html)
//! - [Inverse matrix computation](solve/trait.Inverse.html)
//!
//! Naming Convention
//! -----------------------
//! Each routine is usually exposed as a trait, implemented by the relevant types.
//!
//! For each routine there might be multiple "variants": different traits corresponding to the different ownership possibilities of the array you intend to work on.
//!
//! For example, if you are interested in the QR decomposition of a square matrix, you can use:
//! - [QRSquare](qr/trait.QRSquare.html), if you hold an immutable reference (i.e. `&self`) to the matrix you want to decompose;
//! - [QRSquareInplace](qr/trait.QRSquareInplace.html), if you hold a mutable reference (i.e. `&mut self`) to the matrix you want to decompose;
//! - [QRSquareInto](qr/trait.QRSquareInto.html), if you can pass the matrix you want to decompose by value (e.g. `self`).
//!
//! Depending on the algorithm, each variant might require more or less copy operations of the underlying data.
//!
//! Details are provided in the description of each routine.
//!
//!  Utilities
//!  -----------
//!  - [Assertions for array](index.html#macros)
//!  - [Random matrix generators](generate/index.html)
//!  - [Scalar trait](types/trait.Scalar.html)

#[cfg(features = "openblas")]
extern crate openblas_src;

extern crate blas_src;
extern crate lapack_src;

pub mod assert;
pub mod cholesky;
pub mod convert;
pub mod diagonal;
pub mod eigh;
pub mod error;
pub mod generate;
pub mod inner;
pub mod krylov;
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
pub use inner::*;
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
