//! Solve symmetric linear problem using the Bunch-Kaufman diagonal pivoting method.
//!
//! See also [the manual of dsytrf](http://www.netlib.org/lapack/lapack-3.1.1/html/dsytrf.f.html)

use super::*;
use crate::{error::*, layout::MatrixLayout, types::*};

pub trait Solveh_: Sized {
    /// Bunch-Kaufman: wrapper of `*sytrf` and `*hetrf`
    unsafe fn bk(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<Pivot>;
    /// Wrapper of `*sytri` and `*hetri`
    unsafe fn invh(l: MatrixLayout, uplo: UPLO, a: &mut [Self], ipiv: &Pivot) -> Result<()>;
    /// Wrapper of `*sytrs` and `*hetrs`
    unsafe fn solveh(
        l: MatrixLayout,
        uplo: UPLO,
        a: &[Self],
        ipiv: &Pivot,
        b: &mut [Self],
    ) -> Result<()>;
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
                    $trf(l.lapacke_layout(), uplo as u8, n, a, l.lda(), &mut ipiv)
                        .as_lapack_result()?;
                    Ok(ipiv)
                }
            }

            unsafe fn invh(
                l: MatrixLayout,
                uplo: UPLO,
                a: &mut [Self],
                ipiv: &Pivot,
            ) -> Result<()> {
                let (n, _) = l.size();
                $tri(l.lapacke_layout(), uplo as u8, n, a, l.lda(), ipiv).as_lapack_result()?;
                Ok(())
            }

            unsafe fn solveh(
                l: MatrixLayout,
                uplo: UPLO,
                a: &[Self],
                ipiv: &Pivot,
                b: &mut [Self],
            ) -> Result<()> {
                let (n, _) = l.size();
                let nrhs = 1;
                let ldb = match l {
                    MatrixLayout::C(_) => 1,
                    MatrixLayout::F(_) => n,
                };
                $trs(
                    l.lapacke_layout(),
                    uplo as u8,
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
} // impl_solveh!

impl_solveh!(f64, lapacke::dsytrf, lapacke::dsytri, lapacke::dsytrs);
impl_solveh!(f32, lapacke::ssytrf, lapacke::ssytri, lapacke::ssytrs);
impl_solveh!(c64, lapacke::zhetrf, lapacke::zhetri, lapacke::zhetrs);
impl_solveh!(c32, lapacke::chetrf, lapacke::chetri, lapacke::chetrs);
