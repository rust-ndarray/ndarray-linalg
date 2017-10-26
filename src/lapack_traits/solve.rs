//! Solve linear problem using LU decomposition

use lapack::c;

use error::*;
use layout::MatrixLayout;
use num_traits::Zero;
use types::*;

use super::{Pivot, Transpose, into_result};
use super::opnorm::NormType;

/// Wraps `*getrf`, `*getri`, and `*getrs`
pub trait Solve_: AssociatedReal + Sized {
    /// Computes the LU factorization of a general `m x n` matrix `a` using
    /// partial pivoting with row interchanges.
    ///
    /// If the result matches `Err(LinalgError::Lapack(LapackError {
    /// return_code )) if return_code > 0`, then `U[(return_code-1,
    /// return_code-1)]` is exactly zero. The factorization has been completed,
    /// but the factor `U` is exactly singular, and division by zero will occur
    /// if it is used to solve a system of equations.
    unsafe fn lu(MatrixLayout, a: &mut [Self]) -> Result<Pivot>;
    unsafe fn inv(MatrixLayout, a: &mut [Self], &Pivot) -> Result<()>;
    /// Estimates the the reciprocal of the condition number of the matrix in 1-norm.
    ///
    /// `anorm` should be the 1-norm of the matrix `a`.
    unsafe fn rcond(MatrixLayout, a: &[Self], anorm: Self::Real) -> Result<Self::Real>;
    unsafe fn solve(MatrixLayout, Transpose, a: &[Self], &Pivot, b: &mut [Self]) -> Result<()>;
}

macro_rules! impl_solve {
    ($scalar:ty, $getrf:path, $getri:path, $gecon:path, $getrs:path) => {

impl Solve_ for $scalar {
    unsafe fn lu(l: MatrixLayout, a: &mut [Self]) -> Result<Pivot> {
        let (row, col) = l.size();
        let k = ::std::cmp::min(row, col);
        let mut ipiv = vec![0; k as usize];
        let info = $getrf(l.lapacke_layout(), row, col, a, l.lda(), &mut ipiv);
        into_result(info, ipiv)
    }

    unsafe fn inv(l: MatrixLayout, a: &mut [Self], ipiv: &Pivot) -> Result<()> {
        let (n, _) = l.size();
        let info = $getri(l.lapacke_layout(), n, a, l.lda(), ipiv);
        into_result(info, ())
    }

    unsafe fn rcond(l: MatrixLayout, a: &[Self], anorm: Self::Real) -> Result<Self::Real> {
        let (n, _) = l.size();
        let mut rcond = Self::Real::zero();
        let info = $gecon(l.lapacke_layout(), NormType::One as u8, n, a, l.lda(), anorm, &mut rcond);
        into_result(info, rcond)
    }

    unsafe fn solve(l: MatrixLayout, t: Transpose, a: &[Self], ipiv: &Pivot, b: &mut [Self]) -> Result<()> {
        let (n, _) = l.size();
        let nrhs = 1;
        let ldb = 1;
        let info = $getrs(l.lapacke_layout(), t as u8, n, nrhs, a, l.lda(), ipiv, b, ldb);
        into_result(info, ())
    }
}

}} // impl_solve!

impl_solve!(f64, c::dgetrf, c::dgetri, c::dgecon, c::dgetrs);
impl_solve!(f32, c::sgetrf, c::sgetri, c::sgecon, c::sgetrs);
impl_solve!(c64, c::zgetrf, c::zgetri, c::zgecon, c::zgetrs);
impl_solve!(c32, c::cgetrf, c::cgetri, c::cgecon, c::cgetrs);
