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
                        uplo.as_ptr(),
                        &n,
                        AsPtr::as_mut_ptr(a),
                        &l.lda(),
                        ipiv.as_mut_ptr(),
                        AsPtr::as_mut_ptr(&mut work_size),
                        &(-1),
                        &mut info,
                    )
                };
                info.as_lapack_result()?;

                // actual
                let lwork = work_size[0].to_usize().unwrap();
                let mut work: Vec<Self> = unsafe { vec_uninit(lwork) };
                unsafe {
                    $trf(
                        uplo.as_ptr(),
                        &n,
                        AsPtr::as_mut_ptr(a),
                        &l.lda(),
                        ipiv.as_mut_ptr(),
                        AsPtr::as_mut_ptr(&mut work),
                        &(lwork as i32),
                        &mut info,
                    )
                };
                info.as_lapack_result()?;
                Ok(ipiv)
            }

            fn invh(l: MatrixLayout, uplo: UPLO, a: &mut [Self], ipiv: &Pivot) -> Result<()> {
                let (n, _) = l.size();
                let mut info = 0;
                let mut work: Vec<Self> = unsafe { vec_uninit(n as usize) };
                unsafe {
                    $tri(
                        uplo.as_ptr(),
                        &n,
                        AsPtr::as_mut_ptr(a),
                        &l.lda(),
                        ipiv.as_ptr(),
                        AsPtr::as_mut_ptr(&mut work),
                        &mut info,
                    )
                };
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
                unsafe {
                    $trs(
                        uplo.as_ptr(),
                        &n,
                        &1,
                        AsPtr::as_ptr(a),
                        &l.lda(),
                        ipiv.as_ptr(),
                        AsPtr::as_mut_ptr(b),
                        &n,
                        &mut info,
                    )
                };
                info.as_lapack_result()?;
                Ok(())
            }
        }
    };
} // impl_solveh!

impl_solveh!(
    f64,
    lapack_sys::dsytrf_,
    lapack_sys::dsytri_,
    lapack_sys::dsytrs_
);
impl_solveh!(
    f32,
    lapack_sys::ssytrf_,
    lapack_sys::ssytri_,
    lapack_sys::ssytrs_
);
impl_solveh!(
    c64,
    lapack_sys::zhetrf_,
    lapack_sys::zhetri_,
    lapack_sys::zhetrs_
);
impl_solveh!(
    c32,
    lapack_sys::chetrf_,
    lapack_sys::chetri_,
    lapack_sys::chetrs_
);
