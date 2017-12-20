//! Implement linear solver and inverse matrix

use lapacke;

use super::{Transpose, UPLO, into_result};
use error::*;
use layout::MatrixLayout;
use types::*;

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum Diag {
    Unit = b'U',
    NonUnit = b'N',
}

/// Wraps `*trtri` and `*trtrs`
pub trait Triangular_: Sized {
    unsafe fn inv_triangular(l: MatrixLayout, UPLO, Diag, a: &mut [Self]) -> Result<()>;
    unsafe fn solve_triangular(
        al: MatrixLayout,
        bl: MatrixLayout,
        UPLO,
        Diag,
        a: &[Self],
        b: &mut [Self],
    ) -> Result<()>;
}

macro_rules! impl_triangular {
    ($scalar:ty, $trtri:path, $trtrs:path) => {

impl Triangular_ for $scalar {
    unsafe fn inv_triangular(l: MatrixLayout, uplo: UPLO, diag: Diag, a: &mut [Self]) -> Result<()> {
        let (n, _) = l.size();
        let lda = l.lda();
        let info = $trtri(l.lapacke_layout(), uplo as u8, diag as u8, n, a, lda);
        into_result(info, ())
    }

    unsafe fn solve_triangular(al: MatrixLayout, bl: MatrixLayout, uplo: UPLO, diag: Diag, a: &[Self], mut b: &mut [Self]) -> Result<()> {
        let (n, _) = al.size();
        let lda = al.lda();
        let (_, nrhs) = bl.size();
        let ldb = bl.lda();
        let info = $trtrs(al.lapacke_layout(), uplo as u8, Transpose::No as u8, diag as u8, n, nrhs, a, lda, &mut b, ldb);
        into_result(info, ())
    }
}

}} // impl_triangular!

impl_triangular!(f64, lapacke::dtrtri, lapacke::dtrtrs);
impl_triangular!(f32, lapacke::strtri, lapacke::strtrs);
impl_triangular!(c64, lapacke::ztrtri, lapacke::ztrtrs);
impl_triangular!(c32, lapacke::ctrtri, lapacke::ctrtrs);
