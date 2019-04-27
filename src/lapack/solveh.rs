//! Solve symmetric linear problem using the Bunch-Kaufman diagonal pivoting method.
//!
//! See also [the manual of dsytrf](http://www.netlib.org/lapack/lapack-3.1.1/html/dsytrf.f.html)

use lapacke;

use crate::error::*;
use crate::layout::MatrixLayout;
use crate::types::*;

use super::{into_result, Pivot, UPLO};

pub trait Solveh_: Sized {
    /// Bunch-Kaufman: wrapper of `*sytrf` and `*hetrf`
    unsafe fn bk(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<Pivot>;
    /// Wrapper of `*sytri` and `*hetri`
    unsafe fn invh(l: MatrixLayout, uplo: UPLO, a: &mut [Self], ipiv: &Pivot) -> Result<()>;
    /// Wrapper of `*sytrs` and `*hetrs`
    unsafe fn solveh(l: MatrixLayout, uplo: UPLO, a: &[Self], ipiv: &Pivot, b: &mut [Self]) -> Result<()>;
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
                let ldb = match l {
                    MatrixLayout::C(_) => 1,
                    MatrixLayout::F(_) => n,
                };
                let info = $trs(l.lapacke_layout(), uplo as u8, n, nrhs, a, l.lda(), ipiv, b, ldb);
                into_result(info, ())
            }
        }
    };
} // impl_solveh!

impl_solveh!(f64, lapacke::dsytrf, lapacke::dsytri, lapacke::dsytrs);
impl_solveh!(f32, lapacke::ssytrf, lapacke::ssytri, lapacke::ssytrs);
impl_solveh!(c64, lapacke::zhetrf, lapacke::zhetri, lapacke::zhetrs);
impl_solveh!(c32, lapacke::chetrf, lapacke::chetri, lapacke::chetrs);
