//! Cholesky decomposition

use super::*;
use crate::{error::*, layout::MatrixLayout};
use cauchy::*;

pub trait Cholesky_: Sized {
    /// Cholesky: wrapper of `*potrf`
    ///
    /// **Warning: Only the portion of `a` corresponding to `UPLO` is written.**
    fn cholesky(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<()>;

    /// Wrapper of `*potri`
    ///
    /// **Warning: Only the portion of `a` corresponding to `UPLO` is written.**
    fn inv_cholesky(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<()>;

    /// Wrapper of `*potrs`
    fn solve_cholesky(l: MatrixLayout, uplo: UPLO, a: &[Self], b: &mut [Self]) -> Result<()>;
}

macro_rules! impl_cholesky {
    ($scalar:ty, $trf:path, $tri:path, $trs:path) => {
        impl Cholesky_ for $scalar {
            fn cholesky(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<()> {
                let (n, _) = l.size();
                let mut info = 0;
                let uplo = match l {
                    MatrixLayout::F { .. } => uplo,
                    MatrixLayout::C { .. } => uplo.t(),
                };
                unsafe {
                    $trf(uplo as u8, n, a, n, &mut info);
                }
                info.as_lapack_result()?;
                Ok(())
            }

            fn inv_cholesky(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<()> {
                let (n, _) = l.size();
                let mut info = 0;
                let uplo = match l {
                    MatrixLayout::F { .. } => uplo,
                    MatrixLayout::C { .. } => uplo.t(),
                };
                unsafe {
                    $tri(uplo as u8, n, a, l.lda(), &mut info);
                }
                info.as_lapack_result()?;
                Ok(())
            }

            fn solve_cholesky(
                l: MatrixLayout,
                uplo: UPLO,
                a: &[Self],
                b: &mut [Self],
            ) -> Result<()> {
                let (n, _) = l.size();
                let nrhs = 1;
                let uplo = match l {
                    MatrixLayout::F { .. } => uplo,
                    MatrixLayout::C { .. } => uplo.t(),
                };
                let mut info = 0;
                unsafe {
                    $trs(uplo as u8, n, nrhs, a, l.lda(), b, n, &mut info);
                }
                info.as_lapack_result()?;
                Ok(())
            }
        }
    };
} // end macro_rules

impl_cholesky!(f64, lapack::dpotrf, lapack::dpotri, lapack::dpotrs);
impl_cholesky!(f32, lapack::spotrf, lapack::spotri, lapack::spotrs);
impl_cholesky!(c64, lapack::zpotrf, lapack::zpotri, lapack::zpotrs);
impl_cholesky!(c32, lapack::cpotrf, lapack::cpotri, lapack::cpotrs);
