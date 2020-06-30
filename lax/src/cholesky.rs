//! Cholesky decomposition

use super::*;
use crate::{error::*, layout::MatrixLayout};
use cauchy::*;

pub trait Cholesky_: Sized {
    /// Cholesky: wrapper of `*potrf`
    ///
    /// **Warning: Only the portion of `a` corresponding to `UPLO` is written.**
    unsafe fn cholesky(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<()>;
    /// Wrapper of `*potri`
    ///
    /// **Warning: Only the portion of `a` corresponding to `UPLO` is written.**
    unsafe fn inv_cholesky(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<()>;
    /// Wrapper of `*potrs`
    unsafe fn solve_cholesky(l: MatrixLayout, uplo: UPLO, a: &[Self], b: &mut [Self])
        -> Result<()>;
}

macro_rules! impl_cholesky {
    ($scalar:ty, $trf:path, $tri:path, $trs:path) => {
        impl Cholesky_ for $scalar {
            unsafe fn cholesky(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<()> {
                let (n, _) = l.size();
                $trf(l.lapacke_layout(), uplo as u8, n, a, n).as_lapack_result()?;
                Ok(())
            }

            unsafe fn inv_cholesky(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<()> {
                let (n, _) = l.size();
                $tri(l.lapacke_layout(), uplo as u8, n, a, l.lda()).as_lapack_result()?;
                Ok(())
            }

            unsafe fn solve_cholesky(
                l: MatrixLayout,
                uplo: UPLO,
                a: &[Self],
                b: &mut [Self],
            ) -> Result<()> {
                let (n, _) = l.size();
                let nrhs = 1;
                let ldb = 1;
                $trs(l.lapacke_layout(), uplo as u8, n, nrhs, a, l.lda(), b, ldb)
                    .as_lapack_result()?;
                Ok(())
            }
        }
    };
} // end macro_rules

impl_cholesky!(f64, lapacke::dpotrf, lapacke::dpotri, lapacke::dpotrs);
impl_cholesky!(f32, lapacke::spotrf, lapacke::spotri, lapacke::spotrs);
impl_cholesky!(c64, lapacke::zpotrf, lapacke::zpotri, lapacke::zpotrs);
impl_cholesky!(c32, lapacke::cpotrf, lapacke::cpotri, lapacke::cpotrs);
