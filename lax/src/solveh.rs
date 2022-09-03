use crate::{error::*, layout::MatrixLayout, *};
use cauchy::*;
use num_traits::{ToPrimitive, Zero};

#[cfg_attr(doc, katexit::katexit)]
/// Solve symmetric/hermite indefinite linear problem using the [Bunch-Kaufman diagonal pivoting method][BK].
///
/// For a given symmetric matrix $A$,
/// this method factorizes $A = U^T D U$ or $A = L D L^T$ where
///
/// - $U$ (or $L$) are is a product of permutation and unit upper (lower) triangular matrices
/// - $D$ is symmetric and block diagonal with 1-by-1 and 2-by-2 diagonal blocks.
///
/// This takes two-step approach based in LAPACK:
///
/// 1. Factorize given matrix $A$ into upper ($U$) or lower ($L$) form with diagonal matrix $D$
/// 2. Then solve linear equation $Ax = b$, and/or calculate inverse matrix $A^{-1}$
///
/// [BK]: https://doi.org/10.2307/2005787
///
pub trait Solveh_: Sized {
    /// Factorize input matrix using Bunch-Kaufman diagonal pivoting method
    ///
    /// LAPACK correspondance
    /// ----------------------
    ///
    /// | f32      | f64      | c32      | c64      |
    /// |:---------|:---------|:---------|:---------|
    /// | [ssytrf] | [dsytrf] | [chetrf] | [zhetrf] |
    ///
    /// [ssytrf]: https://netlib.org/lapack/explore-html/d0/d14/group__real_s_ycomputational_ga12d2e56511cf7df066712c61d9acec45.html
    /// [dsytrf]: https://netlib.org/lapack/explore-html/d3/db6/group__double_s_ycomputational_gad91bde1212277b3e909eb6af7f64858a.html
    /// [chetrf]: https://netlib.org/lapack/explore-html/d4/d74/group__complex_h_ecomputational_ga081dd1908e46d064c2bf0a1f6b664b86.html
    /// [zhetrf]: https://netlib.org/lapack/explore-html/d3/d80/group__complex16_h_ecomputational_gadc84a5c9818ee12ea19944623131bd52.html
    ///
    fn bk(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<Pivot>;

    /// Compute inverse matrix $A^{-1}$ from factroized result
    ///
    /// LAPACK correspondance
    /// ----------------------
    ///
    /// | f32      | f64      | c32      | c64      |
    /// |:---------|:---------|:---------|:---------|
    /// | [ssytri] | [dsytri] | [chetri] | [zhetri] |
    ///
    /// [ssytri]: https://netlib.org/lapack/explore-html/d0/d14/group__real_s_ycomputational_gaef378ec0761234aac417f487b43b7a8b.html
    /// [dsytri]: https://netlib.org/lapack/explore-html/d3/db6/group__double_s_ycomputational_ga75e09b4299b7955044a3bbf84c46b593.html
    /// [chetri]: https://netlib.org/lapack/explore-html/d4/d74/group__complex_h_ecomputational_gad87a6a1ac131c5635d47ac440e672bcf.html
    /// [zhetri]: https://netlib.org/lapack/explore-html/d3/d80/group__complex16_h_ecomputational_ga4d9cfa0653de400029b8051996ce1e96.html
    ///
    fn invh(l: MatrixLayout, uplo: UPLO, a: &mut [Self], ipiv: &Pivot) -> Result<()>;

    /// Solve linear equation $Ax = b$ using factroized result
    ///
    /// LAPACK correspondance
    /// ----------------------
    ///
    /// | f32      | f64      | c32      | c64      |
    /// |:---------|:---------|:---------|:---------|
    /// | [ssytrs] | [dsytrs] | [chetrs] | [zhetrs] |
    ///
    /// [ssytrs]: https://netlib.org/lapack/explore-html/d0/d14/group__real_s_ycomputational_gae20133a1119b69a7319783ff982c8c62.html
    /// [dsytrs]: https://netlib.org/lapack/explore-html/d3/db6/group__double_s_ycomputational_ga6a223e61effac7232e98b422f147f032.html
    /// [chetrs]: https://netlib.org/lapack/explore-html/d4/d74/group__complex_h_ecomputational_ga6f9d8da222ffaa7b7535efc922faa1dc.html
    /// [zhetrs]: https://netlib.org/lapack/explore-html/d3/d80/group__complex16_h_ecomputational_gacf697e3bb72c5fd88cd90972999401dd.html
    ///
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
                        AsPtr::as_mut_ptr(&mut ipiv),
                        AsPtr::as_mut_ptr(&mut work_size),
                        &(-1),
                        &mut info,
                    )
                };
                info.as_lapack_result()?;

                // actual
                let lwork = work_size[0].to_usize().unwrap();
                let mut work: Vec<MaybeUninit<Self>> = unsafe { vec_uninit(lwork) };
                unsafe {
                    $trf(
                        uplo.as_ptr(),
                        &n,
                        AsPtr::as_mut_ptr(a),
                        &l.lda(),
                        AsPtr::as_mut_ptr(&mut ipiv),
                        AsPtr::as_mut_ptr(&mut work),
                        &(lwork as i32),
                        &mut info,
                    )
                };
                info.as_lapack_result()?;
                let ipiv = unsafe { ipiv.assume_init() };
                Ok(ipiv)
            }

            fn invh(l: MatrixLayout, uplo: UPLO, a: &mut [Self], ipiv: &Pivot) -> Result<()> {
                let (n, _) = l.size();
                let mut info = 0;
                let mut work: Vec<MaybeUninit<Self>> = unsafe { vec_uninit(n as usize) };
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
