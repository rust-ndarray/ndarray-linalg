//! Solve linear problem using LU decomposition

use lapack::c;

use error::*;
use layout::MatrixLayout;
use types::*;

use super::{Pivot, Transpose, into_result};

/// Wraps `*getrf`, `*getri`, and `*getrs`
pub trait Solve_: Sized {
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
    unsafe fn solve(MatrixLayout, Transpose, a: &[Self], &Pivot, b: &mut [Self]) -> Result<()>;
}

macro_rules! impl_solve {
    ($scalar:ty, $getrf:path, $getri:path, $getrs:path) => {

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

    unsafe fn solve(l: MatrixLayout, t: Transpose, a: &[Self], ipiv: &Pivot, b: &mut [Self]) -> Result<()> {
        let (n, _) = l.size();
        let nrhs = 1;
        let ldb = 1;
        let info = $getrs(l.lapacke_layout(), t as u8, n, nrhs, a, l.lda(), ipiv, b, ldb);
        into_result(info, ())
    }
}

}} // impl_solve!

impl_solve!(f64, c::dgetrf, c::dgetri, c::dgetrs);
impl_solve!(f32, c::sgetrf, c::sgetri, c::sgetrs);
impl_solve!(c64, c::zgetrf, c::zgetri, c::zgetrs);
impl_solve!(c32, c::cgetrf, c::cgetri, c::cgetrs);
