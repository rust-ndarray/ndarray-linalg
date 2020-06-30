//! Solve linear problem using LU decomposition

use super::*;
use crate::{error::*, layout::MatrixLayout, types::*};
use num_traits::Zero;

/// Wraps `*getrf`, `*getri`, and `*getrs`
pub trait Solve_: Scalar + Sized {
    /// Computes the LU factorization of a general `m x n` matrix `a` using
    /// partial pivoting with row interchanges.
    ///
    /// If the result matches `Err(LinalgError::Lapack(LapackError {
    /// return_code )) if return_code > 0`, then `U[(return_code-1,
    /// return_code-1)]` is exactly zero. The factorization has been completed,
    /// but the factor `U` is exactly singular, and division by zero will occur
    /// if it is used to solve a system of equations.
    unsafe fn lu(l: MatrixLayout, a: &mut [Self]) -> Result<Pivot>;
    unsafe fn inv(l: MatrixLayout, a: &mut [Self], p: &Pivot) -> Result<()>;
    /// Estimates the the reciprocal of the condition number of the matrix in 1-norm.
    ///
    /// `anorm` should be the 1-norm of the matrix `a`.
    unsafe fn rcond(l: MatrixLayout, a: &[Self], anorm: Self::Real) -> Result<Self::Real>;
    unsafe fn solve(
        l: MatrixLayout,
        t: Transpose,
        a: &[Self],
        p: &Pivot,
        b: &mut [Self],
    ) -> Result<()>;
}

macro_rules! impl_solve {
    ($scalar:ty, $getrf:path, $getri:path, $gecon:path, $getrs:path) => {
        impl Solve_ for $scalar {
            unsafe fn lu(l: MatrixLayout, a: &mut [Self]) -> Result<Pivot> {
                let (row, col) = l.size();
                let k = ::std::cmp::min(row, col);
                let mut ipiv = vec![0; k as usize];
                $getrf(l.lapacke_layout(), row, col, a, l.lda(), &mut ipiv).as_lapack_result()?;
                Ok(ipiv)
            }

            unsafe fn inv(l: MatrixLayout, a: &mut [Self], ipiv: &Pivot) -> Result<()> {
                let (n, _) = l.size();
                $getri(l.lapacke_layout(), n, a, l.lda(), ipiv).as_lapack_result()?;
                Ok(())
            }

            unsafe fn rcond(l: MatrixLayout, a: &[Self], anorm: Self::Real) -> Result<Self::Real> {
                let (n, _) = l.size();
                let mut rcond = Self::Real::zero();
                $gecon(
                    l.lapacke_layout(),
                    NormType::One as u8,
                    n,
                    a,
                    l.lda(),
                    anorm,
                    &mut rcond,
                )
                .as_lapack_result()?;
                Ok(rcond)
            }

            unsafe fn solve(
                l: MatrixLayout,
                t: Transpose,
                a: &[Self],
                ipiv: &Pivot,
                b: &mut [Self],
            ) -> Result<()> {
                let (n, _) = l.size();
                let nrhs = 1;
                let ldb = 1;
                $getrs(
                    l.lapacke_layout(),
                    t as u8,
                    n,
                    nrhs,
                    a,
                    l.lda(),
                    ipiv,
                    b,
                    ldb,
                )
                .as_lapack_result()?;
                Ok(())
            }
        }
    };
} // impl_solve!

impl_solve!(
    f64,
    lapacke::dgetrf,
    lapacke::dgetri,
    lapacke::dgecon,
    lapacke::dgetrs
);
impl_solve!(
    f32,
    lapacke::sgetrf,
    lapacke::sgetri,
    lapacke::sgecon,
    lapacke::sgetrs
);
impl_solve!(
    c64,
    lapacke::zgetrf,
    lapacke::zgetri,
    lapacke::zgecon,
    lapacke::zgetrs
);
impl_solve!(
    c32,
    lapacke::cgetrf,
    lapacke::cgetri,
    lapacke::cgecon,
    lapacke::cgetrs
);
