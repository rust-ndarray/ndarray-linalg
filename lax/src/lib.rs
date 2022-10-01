//! Safe Rust wrapper for LAPACK without external dependency.
//!
//! [Lapack] trait
//! ----------------
//!
//! This crates provides LAPACK wrapper as a traits.
//! For example, LU decomposition of general matrices is provided like:
//!
//! ```
//! pub trait Lapack{
//!     fn lu(l: MatrixLayout, a: &mut [Self]) -> Result<Pivot>;
//! }
//! ```
//!
//! see [Lapack] for detail.
//! This trait is implemented for [f32], [f64], [c32] which is an alias to `num::Complex<f32>`,
//! and [c64] which is an alias to `num::Complex<f64>`.
//! You can use it like `f64::lu`:
//!
//! ```
//! use lax::{Lapack, layout::MatrixLayout, Transpose};
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
//! use lax::{Lapack, layout::MatrixLayout, Transpose};
//!
//! fn solve_at_once<T: Lapack>(layout: MatrixLayout, a: &mut [T], b: &mut [T]) -> Result<(), lax::error::Error> {
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
//! - [solve] module provides methods for LU-decomposition for general matrix.
//! - [solveh] module provides methods for Bunch-Kaufman diagonal pivoting method for symmetric/hermite indefinite matrix.
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
//! - [svd] module for singular value decomposition (SVD) for general matrix
//! - [svddc] module for singular value decomposition (SVD) with divided-and-conquer algorithm for general matrix
//! - [least_squares] module for solving least square problem using SVD
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
pub mod least_squares;
pub mod qr;
pub mod solve;
pub mod solveh;
pub mod svd;
pub mod svddc;

mod alloc;
mod cholesky;
mod opnorm;
mod rcond;
mod triangular;
mod tridiagonal;

pub use self::cholesky::*;
pub use self::flags::*;
pub use self::least_squares::LeastSquaresOwned;
pub use self::opnorm::*;
pub use self::rcond::*;
pub use self::svd::{SvdOwned, SvdRef};
pub use self::triangular::*;
pub use self::tridiagonal::*;

use self::{alloc::*, error::*, layout::*};
use cauchy::*;
use std::mem::MaybeUninit;

pub type Pivot = Vec<i32>;

#[cfg_attr(doc, katexit::katexit)]
/// Trait for primitive types which implements LAPACK subroutines
pub trait Lapack: OperatorNorm_ + Triangular_ + Tridiagonal_ + Rcond_ {
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

    /// Compute singular-value decomposition (SVD)
    fn svd(l: MatrixLayout, calc_u: bool, calc_vt: bool, a: &mut [Self]) -> Result<SvdOwned<Self>>;

    /// Compute singular value decomposition (SVD) with divide-and-conquer algorithm
    fn svddc(layout: MatrixLayout, jobz: JobSvd, a: &mut [Self]) -> Result<SvdOwned<Self>>;

    /// Compute a vector $x$ which minimizes Euclidian norm $\| Ax - b\|$
    /// for a given matrix $A$ and a vector $b$.
    fn least_squares(
        a_layout: MatrixLayout,
        a: &mut [Self],
        b: &mut [Self],
    ) -> Result<LeastSquaresOwned<Self>>;

    /// Solve least square problems $\argmin_X \| AX - B\|$
    fn least_squares_nrhs(
        a_layout: MatrixLayout,
        a: &mut [Self],
        b_layout: MatrixLayout,
        b: &mut [Self],
    ) -> Result<LeastSquaresOwned<Self>>;

    /// Computes the LU decomposition of a general $m \times n$ matrix
    /// with partial pivoting with row interchanges.
    ///
    /// For a given matrix $A$, LU decomposition is described as $A = PLU$ where:
    ///
    /// - $L$ is lower matrix
    /// - $U$ is upper matrix
    /// - $P$ is permutation matrix represented by [Pivot]
    ///
    /// This is designed as two step computation according to LAPACK API:
    ///
    /// 1. Factorize input matrix $A$ into $L$, $U$, and $P$.
    /// 2. Solve linear equation $Ax = b$ by [Lapack::solve]
    ///    or compute inverse matrix $A^{-1}$ by [Lapack::inv] using the output of LU decomposition.
    ///
    /// Output
    /// -------
    /// - $U$ and $L$ are stored in `a` after LU decomposition has succeeded.
    /// - $P$ is returned as [Pivot]
    ///
    /// Error
    /// ------
    /// - if the matrix is singular
    ///   - On this case, `return_code` in [Error::LapackComputationalFailure] means
    ///     `return_code`-th diagonal element of $U$ becomes zero.
    ///
    fn lu(l: MatrixLayout, a: &mut [Self]) -> Result<Pivot>;

    /// Compute inverse matrix $A^{-1}$ from the output of LU-decomposition
    fn inv(l: MatrixLayout, a: &mut [Self], p: &Pivot) -> Result<()>;

    /// Solve linear equations $Ax = b$ using the output of LU-decomposition
    fn solve(l: MatrixLayout, t: Transpose, a: &[Self], p: &Pivot, b: &mut [Self]) -> Result<()>;

    /// Factorize symmetric/Hermitian matrix using Bunch-Kaufman diagonal pivoting method
    ///
    ///
    /// For a given symmetric matrix $A$,
    /// this method factorizes $A = U^T D U$ or $A = L D L^T$ where
    ///
    /// - $U$ (or $L$) are is a product of permutation and unit upper (lower) triangular matrices
    /// - $D$ is symmetric and block diagonal with 1-by-1 and 2-by-2 diagonal blocks.
    ///
    /// This takes two-step approach based in LAPACK:
    ///
    /// 1. Factorize given matrix $A$ into upper ($U$) or lower ($L$) form with diagonal matrix $D$
    /// 2. Then solve linear equation $Ax = b$, and/or calculate inverse matrix $A^{-1}$
    ///
    /// [BK]: https://doi.org/10.2307/2005787
    ///
    fn bk(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<Pivot>;

    /// Compute inverse matrix $A^{-1}$ of symmetric/Hermitian matrix using factroized result
    fn invh(l: MatrixLayout, uplo: UPLO, a: &mut [Self], ipiv: &Pivot) -> Result<()>;

    /// Solve symmetric/Hermitian linear equation $Ax = b$ using factroized result
    fn solveh(l: MatrixLayout, uplo: UPLO, a: &[Self], ipiv: &Pivot, b: &mut [Self]) -> Result<()>;

    /// Solve symmetric/hermite positive-definite linear equations using Cholesky decomposition
    ///
    /// For a given positive definite matrix $A$,
    /// Cholesky decomposition is described as $A = U^T U$ or $A = LL^T$ where
    ///
    /// - $L$ is lower matrix
    /// - $U$ is upper matrix
    ///
    /// This is designed as two step computation according to LAPACK API
    ///
    /// 1. Factorize input matrix $A$ into $L$ or $U$
    /// 2. Solve linear equation $Ax = b$ or compute inverse matrix $A^{-1}$
    ///    using $U$ or $L$.
    fn cholesky(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<()>;

    /// Compute inverse matrix $A^{-1}$ using $U$ or $L$
    fn inv_cholesky(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<()>;

    /// Solve linear equation $Ax = b$ using $U$ or $L$
    fn solve_cholesky(l: MatrixLayout, uplo: UPLO, a: &[Self], b: &mut [Self]) -> Result<()>;
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

            fn svd(
                l: MatrixLayout,
                calc_u: bool,
                calc_vt: bool,
                a: &mut [Self],
            ) -> Result<SvdOwned<Self>> {
                use svd::*;
                let work = SvdWork::<$s>::new(l, calc_u, calc_vt)?;
                work.eval(a)
            }

            fn svddc(layout: MatrixLayout, jobz: JobSvd, a: &mut [Self]) -> Result<SvdOwned<Self>> {
                use svddc::*;
                let work = SvdDcWork::<$s>::new(layout, jobz)?;
                work.eval(a)
            }

            fn least_squares(
                l: MatrixLayout,
                a: &mut [Self],
                b: &mut [Self],
            ) -> Result<LeastSquaresOwned<Self>> {
                let b_layout = l.resized(b.len() as i32, 1);
                Self::least_squares_nrhs(l, a, b_layout, b)
            }

            fn least_squares_nrhs(
                a_layout: MatrixLayout,
                a: &mut [Self],
                b_layout: MatrixLayout,
                b: &mut [Self],
            ) -> Result<LeastSquaresOwned<Self>> {
                use least_squares::*;
                let work = LeastSquaresWork::<$s>::new(a_layout, b_layout)?;
                work.eval(a, b)
            }

            fn lu(l: MatrixLayout, a: &mut [Self]) -> Result<Pivot> {
                use solve::*;
                LuImpl::lu(l, a)
            }

            fn inv(l: MatrixLayout, a: &mut [Self], p: &Pivot) -> Result<()> {
                use solve::*;
                let mut work = InvWork::<$s>::new(l)?;
                work.calc(a, p)?;
                Ok(())
            }

            fn solve(
                l: MatrixLayout,
                t: Transpose,
                a: &[Self],
                p: &Pivot,
                b: &mut [Self],
            ) -> Result<()> {
                use solve::*;
                SolveImpl::solve(l, t, a, p, b)
            }

            fn bk(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<Pivot> {
                use solveh::*;
                let work = BkWork::<$s>::new(l)?;
                work.eval(uplo, a)
            }

            fn invh(l: MatrixLayout, uplo: UPLO, a: &mut [Self], ipiv: &Pivot) -> Result<()> {
                use solveh::*;
                let mut work = InvhWork::<$s>::new(l)?;
                work.calc(uplo, a, ipiv)
            }

            fn solveh(
                l: MatrixLayout,
                uplo: UPLO,
                a: &[Self],
                ipiv: &Pivot,
                b: &mut [Self],
            ) -> Result<()> {
                use solveh::*;
                SolvehImpl::solveh(l, uplo, a, ipiv, b)
            }

            fn cholesky(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<()> {
                use cholesky::*;
                CholeskyImpl::cholesky(l, uplo, a)
            }

            fn inv_cholesky(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<()> {
                use cholesky::*;
                InvCholeskyImpl::inv_cholesky(l, uplo, a)
            }

            fn solve_cholesky(
                l: MatrixLayout,
                uplo: UPLO,
                a: &[Self],
                b: &mut [Self],
            ) -> Result<()> {
                use cholesky::*;
                SolveCholeskyImpl::solve_cholesky(l, uplo, a, b)
            }
        }
    };
}
impl_lapack!(c64);
impl_lapack!(c32);
impl_lapack!(f64);
impl_lapack!(f32);
