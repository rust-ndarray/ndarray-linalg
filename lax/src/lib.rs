//! Safe Rust wrapper for LAPACK without external dependency.
//!
//! [Lapack] trait
//! ----------------
//!
//! This crates provides LAPACK wrapper as a traits.
//! For example, LU decomposition of general matrices is provided like:
//!
//! ```ignore
//! pub trait Lapack {
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
//! - [solveh] module provides methods for Bunch-Kaufman diagonal pivoting method for symmetric/Hermitian indefinite matrix.
//! - [cholesky] module provides methods for Cholesky decomposition for symmetric/Hermitian positive dinite matrix.
//!
//! Eigenvalue Problem
//! -------------------
//!
//! According to the property input metrix,
//! there are several types of eigenvalue problem API
//!
//! - [eig] module for eigenvalue problem for general matrix.
//! - [eigh] module for eigenvalue problem for symmetric/Hermitian matrix.
//! - [eigh_generalized] module for generalized eigenvalue problem for symmetric/Hermitian matrix.
//!
//! Singular Value Decomposition
//! -----------------------------
//!
//! - [svd] module for singular value decomposition (SVD) for general matrix
//! - [svddc] module for singular value decomposition (SVD) with divided-and-conquer algorithm for general matrix
//! - [least_squares] module for solving least square problem using SVD
//!

#![deny(rustdoc::broken_intra_doc_links, rustdoc::private_intra_doc_links)]

#[cfg(any(feature = "intel-mkl-dynamic-lp64-iomp", feature = "intel-mkl-dynamic-lp64-seq", feature = "intel-mkl-static-ilp64-iomp", feature = "intel-mkl-static-lp64-iomp", feature = "intel-mkl-dynamic-ilp64-iomp", feature = "intel-mkl-static-ilp64-seq", feature = "intel-mkl-static-lp64-seq"))]
extern crate intel_mkl_src as _src;

#[cfg(any(feature = "openblas-system", feature = "openblas-static"))]
extern crate openblas_src as _src;

#[cfg(any(feature = "netlib-system", feature = "netlib-static"))]
extern crate netlib_src as _src;

pub mod alloc;
pub mod cholesky;
pub mod eig;
pub mod eigh;
pub mod eigh_generalized;
pub mod error;
pub mod flags;
pub mod layout;
pub mod least_squares;
pub mod opnorm;
pub mod qr;
pub mod rcond;
pub mod solve;
pub mod solveh;
pub mod svd;
pub mod svddc;
pub mod triangular;
pub mod tridiagonal;

pub use self::flags::*;
pub use self::least_squares::LeastSquaresOwned;
pub use self::svd::{SvdOwned, SvdRef};
pub use self::tridiagonal::{LUFactorizedTridiagonal, Tridiagonal};

use self::{alloc::*, error::*, layout::*};
use cauchy::*;
use std::mem::MaybeUninit;

pub type Pivot = Vec<i32>;

#[cfg_attr(doc, katexit::katexit)]
/// Trait for primitive types which implements LAPACK subroutines
pub trait Lapack: Scalar {
    /// Compute right eigenvalue and eigenvectors for a general matrix
    fn eig(
        calc_v: bool,
        l: MatrixLayout,
        a: &mut [Self],
    ) -> Result<(Vec<Self::Complex>, Vec<Self::Complex>)>;

    /// Compute right eigenvalue and eigenvectors for a symmetric or Hermitian matrix
    fn eigh(
        calc_eigenvec: bool,
        layout: MatrixLayout,
        uplo: UPLO,
        a: &mut [Self],
    ) -> Result<Vec<Self::Real>>;

    /// Compute right eigenvalue and eigenvectors for a symmetric or Hermitian matrix
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
    fn bk(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<Pivot>;

    /// Compute inverse matrix $A^{-1}$ using the result of [Lapack::bk]
    fn invh(l: MatrixLayout, uplo: UPLO, a: &mut [Self], ipiv: &Pivot) -> Result<()>;

    /// Solve symmetric/Hermitian linear equation $Ax = b$ using the result of [Lapack::bk]
    fn solveh(l: MatrixLayout, uplo: UPLO, a: &[Self], ipiv: &Pivot, b: &mut [Self]) -> Result<()>;

    /// Solve symmetric/Hermitian positive-definite linear equations using Cholesky decomposition
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
    /// 2. Solve linear equation $Ax = b$ by [Lapack::solve_cholesky]
    ///    or compute inverse matrix $A^{-1}$ by [Lapack::inv_cholesky]
    ///
    fn cholesky(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<()>;

    /// Compute inverse matrix $A^{-1}$ using $U$ or $L$ calculated by [Lapack::cholesky]
    fn inv_cholesky(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<()>;

    /// Solve linear equation $Ax = b$ using $U$ or $L$ calculated by [Lapack::cholesky]
    fn solve_cholesky(l: MatrixLayout, uplo: UPLO, a: &[Self], b: &mut [Self]) -> Result<()>;

    /// Estimates the the reciprocal of the condition number of the matrix in 1-norm.
    ///
    /// `anorm` should be the 1-norm of the matrix `a`.
    fn rcond(l: MatrixLayout, a: &[Self], anorm: Self::Real) -> Result<Self::Real>;

    /// Compute norm of matrices
    ///
    /// For a $n \times m$ matrix
    /// $$
    /// A = \begin{pmatrix}
    ///   a_{11} & \cdots & a_{1m} \\\\
    ///   \vdots & \ddots & \vdots \\\\
    ///   a_{n1} & \cdots & a_{nm}
    /// \end{pmatrix}
    /// $$
    /// LAPACK can compute three types of norms:
    ///
    /// - Operator norm based on 1-norm for its domain linear space:
    ///   $$
    ///   \Vert A \Vert_1 = \sup_{\Vert x \Vert_1 = 1} \Vert Ax \Vert_1
    ///   = \max_{1 \le j \le m } \sum_{i=1}^n |a_{ij}|
    ///   $$
    ///   where
    ///   $\Vert x\Vert_1 = \sum_{j=1}^m |x_j|$
    ///   is 1-norm for a vector $x$.
    ///
    /// - Operator norm based on $\infty$-norm for its domain linear space:
    ///   $$
    ///   \Vert A \Vert_\infty = \sup_{\Vert x \Vert_\infty = 1} \Vert Ax \Vert_\infty
    ///   = \max_{1 \le i \le n } \sum_{j=1}^m |a_{ij}|
    ///   $$
    ///   where
    ///   $\Vert x\Vert_\infty = \max_{j=1}^m |x_j|$
    ///   is $\infty$-norm for a vector $x$.
    ///
    /// - Frobenious norm
    ///   $$
    ///   \Vert A \Vert_F = \sqrt{\mathrm{Tr} \left(AA^\dagger\right)} = \sqrt{\sum_{i=1}^n \sum_{j=1}^m |a_{ij}|^2}
    ///   $$
    ///
    fn opnorm(t: NormType, l: MatrixLayout, a: &[Self]) -> Self::Real;

    fn solve_triangular(
        al: MatrixLayout,
        bl: MatrixLayout,
        uplo: UPLO,
        d: Diag,
        a: &[Self],
        b: &mut [Self],
    ) -> Result<()>;

    /// Computes the LU factorization of a tridiagonal `m x n` matrix `a` using
    /// partial pivoting with row interchanges.
    fn lu_tridiagonal(a: Tridiagonal<Self>) -> Result<LUFactorizedTridiagonal<Self>>;

    fn rcond_tridiagonal(lu: &LUFactorizedTridiagonal<Self>) -> Result<Self::Real>;

    fn solve_tridiagonal(
        lu: &LUFactorizedTridiagonal<Self>,
        bl: MatrixLayout,
        t: Transpose,
        b: &mut [Self],
    ) -> Result<()>;
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

            fn rcond(l: MatrixLayout, a: &[Self], anorm: Self::Real) -> Result<Self::Real> {
                use rcond::*;
                let mut work = RcondWork::<$s>::new(l);
                work.calc(a, anorm)
            }

            fn opnorm(t: NormType, l: MatrixLayout, a: &[Self]) -> Self::Real {
                use opnorm::*;
                let mut work = OperatorNormWork::<$s>::new(t, l);
                work.calc(a)
            }

            fn solve_triangular(
                al: MatrixLayout,
                bl: MatrixLayout,
                uplo: UPLO,
                d: Diag,
                a: &[Self],
                b: &mut [Self],
            ) -> Result<()> {
                use triangular::*;
                SolveTriangularImpl::solve_triangular(al, bl, uplo, d, a, b)
            }

            fn lu_tridiagonal(a: Tridiagonal<Self>) -> Result<LUFactorizedTridiagonal<Self>> {
                use tridiagonal::*;
                let work = LuTridiagonalWork::<$s>::new(a.l);
                work.eval(a)
            }

            fn rcond_tridiagonal(lu: &LUFactorizedTridiagonal<Self>) -> Result<Self::Real> {
                use tridiagonal::*;
                let mut work = RcondTridiagonalWork::<$s>::new(lu.a.l);
                work.calc(lu)
            }

            fn solve_tridiagonal(
                lu: &LUFactorizedTridiagonal<Self>,
                bl: MatrixLayout,
                t: Transpose,
                b: &mut [Self],
            ) -> Result<()> {
                use tridiagonal::*;
                SolveTridiagonalImpl::solve_tridiagonal(lu, bl, t, b)
            }
        }
    };
}
impl_lapack!(c64);
impl_lapack!(c32);
impl_lapack!(f64);
impl_lapack!(f32);
