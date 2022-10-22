//! The `ndarray-linalg` crate provides linear algebra functionalities for `ArrayBase`, the n-dimensional array data structure provided by [`ndarray`](https://github.com/rust-ndarray/ndarray).
//!
//! `ndarray-linalg` leverages [LAPACK](http://www.netlib.org/lapack/)'s routines using the bindings provided by [blas-lapack-rs/lapack](https://github.com/blas-lapack-rs/lapack).
//!
//! Linear algebra methods
//! -----------------------
//! - Decomposition methods:
//!     - [QR decomposition](qr/index.html)
//!     - [Cholesky/LU decomposition](cholesky/index.html)
//!     - [Eigenvalue decomposition](eig/index.html)
//!     - [Eigenvalue decomposition for Hermite matrices](eigh/index.html)
//!     - [**S**ingular **V**alue **D**ecomposition](svd/index.html)
//! - Solution of linear systems:
//!    - [General matrices](solve/index.html)
//!    - [Triangular matrices](triangular/index.html)
//!    - [Hermitian/real symmetric matrices](solveh/index.html)
//!    - [Tridiagonal matrices](tridiagonal/index.html)
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

#![allow(
    clippy::module_inception,
    clippy::many_single_char_names,
    clippy::type_complexity,
    clippy::ptr_arg
)]
#![deny(rustdoc::broken_intra_doc_links, rustdoc::private_intra_doc_links)]

#[macro_use]
extern crate ndarray;

pub mod assert;
pub mod cholesky;
pub mod convert;
pub mod diagonal;
pub mod eig;
pub mod eigh;
pub mod error;
pub mod expm;
pub mod generate;
pub mod inner;
pub mod krylov;
pub mod layout;
pub mod least_squares;
pub mod lobpcg;
pub mod norm;
pub mod normest1;
pub mod operator;
pub mod opnorm;
pub mod qr;
pub mod solve;
pub mod solveh;
pub mod svd;
pub mod svddc;
pub mod trace;
pub mod triangular;
pub mod tridiagonal;
pub mod types;

pub use crate::assert::*;
pub use crate::cholesky::*;
pub use crate::convert::*;
pub use crate::diagonal::*;
pub use crate::eig::*;
pub use crate::eigh::*;
pub use crate::generate::*;
pub use crate::inner::*;
pub use crate::layout::*;
pub use crate::least_squares::*;
pub use crate::lobpcg::{TruncatedEig, TruncatedOrder, TruncatedSvd};
pub use crate::norm::*;
pub use crate::operator::*;
pub use crate::opnorm::*;
pub use crate::qr::*;
pub use crate::solve::*;
pub use crate::solveh::*;
pub use crate::svd::*;
pub use crate::svddc::*;
pub use crate::trace::*;
pub use crate::triangular::*;
pub use crate::tridiagonal::*;
pub use crate::types::*;
// pub use crate::expm::*;
