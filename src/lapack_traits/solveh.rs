//! Solve symmetric linear problem using the Bunch-Kaufman diagonal pivoting method.
//!
//! See also [the manual of dsytrf](http://www.netlib.org/lapack/lapack-3.1.1/html/dsytrf.f.html)

use lapack::c;

use error::*;
use layout::MatrixLayout;
use types::*;

use super::{Pivot, UPLO, into_result};

pub trait Solveh_: Sized {
    /// Bunch-Kaufman: wrapper of `*sytrf` and `*hetrf`
    unsafe fn bk(MatrixLayout, UPLO, a: &mut [Self]) -> Result<Pivot>;
    /// Wrapper of `*sytri` and `*hetri`
    unsafe fn invh(MatrixLayout, UPLO, a: &mut [Self], &Pivot) -> Result<()>;
    /// Wrapper of `*sytrs` and `*hetrs`
    unsafe fn solveh(MatrixLayout, UPLO, a: &[Self], &Pivot, b: &mut [Self]) -> Result<()>;
}

macro_rules! impl_solveh {
    ($scalar:ty, $trf:path, $tri:path, $trs:path) => {

impl Solveh_ for $scalar {
    unsafe fn bk(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<Pivot> {
        let (n, _) = l.size();
        let mut ipiv = vec![0; n as usize];
        if n == 0 {
            // Work around bug in LAPACKE functions.
            Ok(ipiv)
        } else {
            let info = $trf(l.lapacke_layout(), uplo as u8, n, a, l.lda(), &mut ipiv);
            into_result(info, ipiv)
        }
    }

    unsafe fn invh(l: MatrixLayout, uplo: UPLO, a: &mut [Self], ipiv: &Pivot) -> Result<()> {
        let (n, _) = l.size();
        let info = $tri(l.lapacke_layout(), uplo as u8, n, a, l.lda(), ipiv);
        into_result(info, ())
    }

    unsafe fn solveh(l: MatrixLayout, uplo: UPLO, a: &[Self], ipiv: &Pivot, b: &mut [Self]) -> Result<()> {
        let (n, _) = l.size();
        let nrhs = 1;
        let ldb = 1;
        let info = $trs(l.lapacke_layout(), uplo as u8, n, nrhs, a, l.lda(), ipiv, b, ldb);
        into_result(info, ())
    }
}

}} // impl_solveh!

impl_solveh!(f64, c::dsytrf, c::dsytri, c::dsytrs);
impl_solveh!(f32, c::ssytrf, c::ssytri, c::ssytrs);
impl_solveh!(c64, c::zhetrf, c::zhetri, c::zhetrs);
impl_solveh!(c32, c::chetrf, c::chetri, c::chetrs);
