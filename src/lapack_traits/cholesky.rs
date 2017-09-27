//! Cholesky decomposition

use lapack::c;

use error::*;
use layout::MatrixLayout;
use types::*;

use super::{UPLO, into_result};

pub trait Cholesky_: Sized {
    /// Cholesky: wrapper of `*potrf`
    ///
    /// **Warning: Only the portion of `a` corresponding to `UPLO` is written.**
    unsafe fn cholesky(MatrixLayout, UPLO, a: &mut [Self]) -> Result<()>;
    /// Wrapper of `*potri`
    ///
    /// **Warning: Only the portion of `a` corresponding to `UPLO` is written.**
    unsafe fn inv_cholesky(MatrixLayout, UPLO, a: &mut [Self]) -> Result<()>;
    /// Wrapper of `*potrs`
    unsafe fn solve_cholesky(MatrixLayout, UPLO, a: &[Self], b: &mut [Self]) -> Result<()>;
}

macro_rules! impl_cholesky {
    ($scalar:ty, $trf:path, $tri:path, $trs:path) => {
impl Cholesky_ for $scalar {
    unsafe fn cholesky(l: MatrixLayout, uplo: UPLO, mut a: &mut [Self]) -> Result<()> {
        let (n, _) = l.size();
        let info = $trf(l.lapacke_layout(), uplo as u8, n, a, n);
        into_result(info, ())
    }

    unsafe fn inv_cholesky(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<()> {
        let (n, _) = l.size();
        let info = $tri(l.lapacke_layout(), uplo as u8, n, a, l.lda());
        into_result(info, ())
    }

    unsafe fn solve_cholesky(l: MatrixLayout, uplo: UPLO, a: &[Self], b: &mut [Self]) -> Result<()> {
        let (n, _) = l.size();
        let nrhs = 1;
        let ldb = 1;
        let info = $trs(l.lapacke_layout(), uplo as u8, n, nrhs, a, l.lda(), b, ldb);
        into_result(info, ())
    }
}
}} // end macro_rules

impl_cholesky!(f64, c::dpotrf, c::dpotri, c::dpotrs);
impl_cholesky!(f32, c::spotrf, c::spotri, c::spotrs);
impl_cholesky!(c64, c::zpotrf, c::zpotri, c::zpotrs);
impl_cholesky!(c32, c::cpotrf, c::cpotri, c::cpotrs);
