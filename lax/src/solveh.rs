//! Solve symmetric linear problem using the Bunch-Kaufman diagonal pivoting method.
//!
//! See also [the manual of dsytrf](http://www.netlib.org/lapack/lapack-3.1.1/html/dsytrf.f.html)

use crate::{error::*, layout::MatrixLayout, *};
use cauchy::*;
use num_traits::{ToPrimitive, Zero};

pub trait Solveh_: Sized {
    /// Bunch-Kaufman: wrapper of `*sytrf` and `*hetrf`
    fn bk(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<Pivot>;
    /// Wrapper of `*sytri` and `*hetri`
    fn invh(l: MatrixLayout, uplo: UPLO, a: &mut [Self], ipiv: &Pivot) -> Result<()>;
    /// Wrapper of `*sytrs` and `*hetrs`
    fn solveh(l: MatrixLayout, uplo: UPLO, a: &[Self], ipiv: &Pivot, b: &mut [Self]) -> Result<()>;
}

macro_rules! impl_solveh {
    ($scalar:ty, $trf:path, $tri:path, $trs:path) => {
        impl Solveh_ for $scalar {
            fn bk(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<Pivot> {
                let (n, _) = l.size();
                let mut ipiv = unsafe { vec_uninit(n as usize) };
                if n == 0 {
                    return Ok(Vec::new());
                }

                // calc work size
                let mut info = 0;
                let mut work_size = [Self::zero()];
                unsafe {
                    $trf(
                        uplo as u8,
                        n,
                        a,
                        l.lda(),
                        &mut ipiv,
                        &mut work_size,
                        -1,
                        &mut info,
                    )
                };
                info.as_lapack_result()?;

                // actual
                let lwork = work_size[0].to_usize().unwrap();
                let mut work = unsafe { vec_uninit(lwork) };
                unsafe {
                    $trf(
                        uplo as u8,
                        n,
                        a,
                        l.lda(),
                        &mut ipiv,
                        &mut work,
                        lwork as i32,
                        &mut info,
                    )
                };
                info.as_lapack_result()?;
                Ok(ipiv)
            }

            fn invh(l: MatrixLayout, uplo: UPLO, a: &mut [Self], ipiv: &Pivot) -> Result<()> {
                let (n, _) = l.size();
                let mut info = 0;
                let mut work = unsafe { vec_uninit(n as usize) };
                unsafe { $tri(uplo as u8, n, a, l.lda(), ipiv, &mut work, &mut info) };
                info.as_lapack_result()?;
                Ok(())
            }

            fn solveh(
                l: MatrixLayout,
                uplo: UPLO,
                a: &[Self],
                ipiv: &Pivot,
                b: &mut [Self],
            ) -> Result<()> {
                let (n, _) = l.size();
                let mut info = 0;
                unsafe { $trs(uplo as u8, n, 1, a, l.lda(), ipiv, b, n, &mut info) };
                info.as_lapack_result()?;
                Ok(())
            }
        }
    };
} // impl_solveh!

impl_solveh!(f64, lapack::dsytrf, lapack::dsytri, lapack::dsytrs);
impl_solveh!(f32, lapack::ssytrf, lapack::ssytri, lapack::ssytrs);
impl_solveh!(c64, lapack::zhetrf, lapack::zhetri, lapack::zhetrs);
impl_solveh!(c32, lapack::chetrf, lapack::chetri, lapack::chetrs);
