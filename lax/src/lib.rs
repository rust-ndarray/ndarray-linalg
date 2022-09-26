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
//! - [eig] module for eigenvalue problem for general matrix.
//! - [eigh] module for eigenvalue problem for symmetric/hermite matrix.
//! - [eigh_generalized] module for generalized eigenvalue problem for symmetric/hermite matrix.
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

#![deny(rustdoc::broken_intra_doc_links, rustdoc::private_intra_doc_links)]

#[cfg(any(feature = "intel-mkl-system", feature = "intel-mkl-static"))]
extern crate intel_mkl_src as _src;

#[cfg(any(feature = "openblas-system", feature = "openblas-static"))]
extern crate openblas_src as _src;

#[cfg(any(feature = "netlib-system", feature = "netlib-static"))]
extern crate netlib_src as _src;

pub mod error;
pub mod flags;
pub mod layout;

pub mod eig;
pub mod eigh;
pub mod eigh_generalized;
pub mod qr;

mod alloc;
mod cholesky;
mod least_squares;
mod opnorm;
mod rcond;
mod solve;
mod solveh;
mod svd;
mod svddc;
mod triangular;
mod tridiagonal;

pub use self::cholesky::*;
pub use self::flags::*;
pub use self::least_squares::*;
pub use self::opnorm::*;
pub use self::rcond::*;
pub use self::solve::*;
pub use self::solveh::*;
pub use self::svd::*;
pub use self::svddc::*;
pub use self::triangular::*;
pub use self::tridiagonal::*;

use self::{alloc::*, error::*, layout::*};
use cauchy::*;
use std::mem::MaybeUninit;

pub type Pivot = Vec<i32>;

/// Trait for primitive types which implements LAPACK subroutines
pub trait Lapack:
    OperatorNorm_
    + SVD_
    + SVDDC_
    + Solve_
    + Solveh_
    + Cholesky_
    + Triangular_
    + Tridiagonal_
    + Rcond_
    + LeastSquaresSvdDivideConquer_
{
    /// Compute right eigenvalue and eigenvectors for a general matrix
    fn eig(
        calc_v: bool,
        l: MatrixLayout,
        a: &mut [Self],
    ) -> Result<(Vec<Self::Complex>, Vec<Self::Complex>)>;

    /// Compute right eigenvalue and eigenvectors for a symmetric or hermite matrix
    fn eigh(
        calc_eigenvec: bool,
        layout: MatrixLayout,
        uplo: UPLO,
        a: &mut [Self],
    ) -> Result<Vec<Self::Real>>;

    /// Compute right eigenvalue and eigenvectors for a symmetric or hermite matrix
    fn eigh_generalized(
        calc_eigenvec: bool,
        layout: MatrixLayout,
        uplo: UPLO,
        a: &mut [Self],
        b: &mut [Self],
    ) -> Result<Vec<Self::Real>>;

    /// Execute Householder reflection as the first step of QR-decomposition
    ///
    /// For C-continuous array,
    /// this will call LQ-decomposition of the transposed matrix $ A^T = LQ^T $
    fn householder(l: MatrixLayout, a: &mut [Self]) -> Result<Vec<Self>>;

    /// Reconstruct Q-matrix from Householder-reflectors
    fn q(l: MatrixLayout, a: &mut [Self], tau: &[Self]) -> Result<()>;

    /// Execute QR-decomposition at once
    fn qr(l: MatrixLayout, a: &mut [Self]) -> Result<Vec<Self>>;
}

macro_rules! impl_lapack {
    ($s:ty) => {
        impl Lapack for $s {
            fn eig(
                calc_v: bool,
                l: MatrixLayout,
                a: &mut [Self],
            ) -> Result<(Vec<Self::Complex>, Vec<Self::Complex>)> {
                use eig::*;
                let work = EigWork::<$s>::new(calc_v, l)?;
                let EigOwned { eigs, vr, vl } = work.eval(a)?;
                Ok((eigs, vr.or(vl).unwrap_or_default()))
            }

            fn eigh(
                calc_eigenvec: bool,
                layout: MatrixLayout,
                uplo: UPLO,
                a: &mut [Self],
            ) -> Result<Vec<Self::Real>> {
                use eigh::*;
                let work = EighWork::<$s>::new(calc_eigenvec, layout)?;
                work.eval(uplo, a)
            }

            fn eigh_generalized(
                calc_eigenvec: bool,
                layout: MatrixLayout,
                uplo: UPLO,
                a: &mut [Self],
                b: &mut [Self],
            ) -> Result<Vec<Self::Real>> {
                use eigh_generalized::*;
                let work = EighGeneralizedWork::<$s>::new(calc_eigenvec, layout)?;
                work.eval(uplo, a, b)
            }

            fn householder(l: MatrixLayout, a: &mut [Self]) -> Result<Vec<Self>> {
                use qr::*;
                let work = HouseholderWork::<$s>::new(l)?;
                work.eval(a)
            }

            fn q(l: MatrixLayout, a: &mut [Self], tau: &[Self]) -> Result<()> {
                use qr::*;
                let mut work = QWork::<$s>::new(l)?;
                work.calc(a, tau)?;
                Ok(())
            }

            fn qr(l: MatrixLayout, a: &mut [Self]) -> Result<Vec<Self>> {
                let tau = Self::householder(l, a)?;
                let r = Vec::from(&*a);
                Self::q(l, a, &tau)?;
                Ok(r)
            }
        }
    };
}
impl_lapack!(c64);
impl_lapack!(c32);
impl_lapack!(f64);
impl_lapack!(f32);
