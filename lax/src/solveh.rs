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
    /// | f32    | f64    | c32    | c64    |
    /// |:-------|:-------|:-------|:-------|
    /// | ssytrf | dsytrf | chetrf | zhetrf |
    ///
    fn bk(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<Pivot>;

    /// Compute inverse matrix $A^{-1}$ from factroized result
    ///
    /// LAPACK correspondance
    /// ----------------------
    ///
    /// | f32    | f64    | c32    | c64    |
    /// |:-------|:-------|:-------|:-------|
    /// | ssytri | dsytri | chetri | zhetri |
    ///
    fn invh(l: MatrixLayout, uplo: UPLO, a: &mut [Self], ipiv: &Pivot) -> Result<()>;

    /// Solve linear equation $Ax = b$ using factroized result
    ///
    /// LAPACK correspondance
    /// ----------------------
    ///
    /// | f32    | f64    | c32    | c64    |
    /// |:-------|:-------|:-------|:-------|
    /// | ssytrs | dsytrs | chetrs | zhetrs |
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
