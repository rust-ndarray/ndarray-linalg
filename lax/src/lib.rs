//! ndarray-free safe Rust wrapper for LAPACK FFI
//!
//! `Lapack` trait and sub-traits
//! -------------------------------
//!
//! This crates provides LAPACK wrapper as `impl` of traits to base scalar types.
//! For example, LU decomposition to double-precision matrix is provided like:
//!
//! ```ignore
//! impl Solve_ for f64 {
//!     fn lu(l: MatrixLayout, a: &mut [Self]) -> Result<Pivot> { ... }
//! }
//! ```
//!
//! see [Solve_] for detail. You can use it like `f64::lu`:
//!
//! ```
//! use lax::{Solve_, layout::MatrixLayout, Transpose};
//!
//! let mut a = vec![
//!   1.0, 2.0,
//!   3.0, 4.0
//! ];
//! let mut b = vec![1.0, 2.0];
//! let layout = MatrixLayout::C { row: 2, lda: 2 };
//! let pivot = f64::lu(layout, &mut a).unwrap();
//! f64::solve(layout, Transpose::No, &a, &pivot, &mut b).unwrap();
//! ```
//!
//! When you want to write generic algorithm for real and complex matrices,
//! this trait can be used as a trait bound:
//!
//! ```
//! use lax::{Solve_, layout::MatrixLayout, Transpose};
//!
//! fn solve_at_once<T: Solve_>(layout: MatrixLayout, a: &mut [T], b: &mut [T]) -> Result<(), lax::error::Error> {
//!   let pivot = T::lu(layout, a)?;
//!   T::solve(layout, Transpose::No, a, &pivot, b)?;
//!   Ok(())
//! }
//! ```
//!
//! There are several similar traits as described below to keep development easy.
//! They are merged into a single trait, [Lapack].
//!
//! Linear equation, Inverse matrix, Condition number
//! --------------------------------------------------
//!
//! According to the property input metrix, several types of triangular decomposition are used:
//!
//! - [Solve_] trait provides methods for LU-decomposition for general matrix.
//! - [Solveh_] triat provides methods for Bunch-Kaufman diagonal pivoting method for symmetric/hermite indefinite matrix.
//! - [Cholesky_] triat provides methods for Cholesky decomposition for symmetric/hermite positive dinite matrix.
//!
//! Eigenvalue Problem
//! -------------------
//!
//! According to the property input metrix,
//! there are several types of eigenvalue problem API
//!
//! - [Eig_] trait provides methods for eigenvalue problem for general matrix.
//! - [Eigh_] trait provides methods for eigenvalue problem for symmetric/hermite matrix.
//!
//! Singular Value Decomposition
//! -----------------------------
//!
//! - [SVD_] trait provides methods for singular value decomposition for general matrix
//! - [SVDDC_] trait provides methods for singular value decomposition for general matrix
//!   with divided-and-conquer algorithm
//! - [LeastSquaresSvdDivideConquer_] trait provides methods
//!   for solving least square problem by SVD
//!

#[cfg(any(feature = "intel-mkl-system", feature = "intel-mkl-static"))]
extern crate intel_mkl_src as _src;

#[cfg(any(feature = "openblas-system", feature = "openblas-static"))]
extern crate openblas_src as _src;

#[cfg(any(feature = "netlib-system", feature = "netlib-static"))]
extern crate netlib_src as _src;

pub mod error;
pub mod flags;
pub mod layout;

mod alloc;
mod cholesky;
mod eig;
mod eigh;
mod least_squares;
mod opnorm;
mod qr;
mod rcond;
mod solve;
mod solveh;
mod svd;
mod svddc;
mod triangular;
mod tridiagonal;

pub use self::cholesky::*;
pub use self::eig::*;
pub use self::eigh::*;
pub use self::flags::*;
pub use self::least_squares::*;
pub use self::opnorm::*;
pub use self::qr::*;
pub use self::rcond::*;
pub use self::solve::*;
pub use self::solveh::*;
pub use self::svd::*;
pub use self::svddc::*;
pub use self::triangular::*;
pub use self::tridiagonal::*;

use self::alloc::*;
use cauchy::*;
use std::mem::MaybeUninit;

pub type Pivot = Vec<i32>;

/// Trait for primitive types which implements LAPACK subroutines
pub trait Lapack:
    OperatorNorm_
    + QR_
    + SVD_
    + SVDDC_
    + Solve_
    + Solveh_
    + Cholesky_
    + Eig_
    + Eigh_
    + Triangular_
    + Tridiagonal_
    + Rcond_
    + LeastSquaresSvdDivideConquer_
{
}

impl Lapack for f32 {}
impl Lapack for f64 {}
impl Lapack for c32 {}
impl Lapack for c64 {}
