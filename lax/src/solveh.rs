use crate::{error::*, layout::MatrixLayout, *};
use cauchy::*;
use num_traits::{ToPrimitive, Zero};

pub struct BkWork<T: Scalar> {
    pub layout: MatrixLayout,
    pub work: Vec<MaybeUninit<T>>,
    pub ipiv: Vec<MaybeUninit<i32>>,
}

pub trait BkWorkImpl: Sized {
    type Elem: Scalar;
    fn new(l: MatrixLayout) -> Result<Self>;
    fn calc(&mut self, uplo: UPLO, a: &mut [Self::Elem]) -> Result<&[i32]>;
    fn eval(self, uplo: UPLO, a: &mut [Self::Elem]) -> Result<Pivot>;
}

macro_rules! impl_bk_work {
    ($s:ty, $trf:path) => {
        impl BkWorkImpl for BkWork<$s> {
            type Elem = $s;

            fn new(layout: MatrixLayout) -> Result<Self> {
                let (n, _) = layout.size();
                let ipiv = vec_uninit(n as usize);
                let mut info = 0;
                let mut work_size = [Self::Elem::zero()];
                unsafe {
                    $trf(
                        UPLO::Upper.as_ptr(),
                        &n,
                        std::ptr::null_mut(),
                        &layout.lda(),
                        std::ptr::null_mut(),
                        AsPtr::as_mut_ptr(&mut work_size),
                        &(-1),
                        &mut info,
                    )
                };
                info.as_lapack_result()?;
                let lwork = work_size[0].to_usize().unwrap();
                let work = vec_uninit(lwork);
                Ok(BkWork { layout, work, ipiv })
            }

            fn calc(&mut self, uplo: UPLO, a: &mut [Self::Elem]) -> Result<&[i32]> {
                let (n, _) = self.layout.size();
                let lwork = self.work.len().to_i32().unwrap();
                let mut info = 0;
                unsafe {
                    $trf(
                        uplo.as_ptr(),
                        &n,
                        AsPtr::as_mut_ptr(a),
                        &self.layout.lda(),
                        AsPtr::as_mut_ptr(&mut self.ipiv),
                        AsPtr::as_mut_ptr(&mut self.work),
                        &lwork,
                        &mut info,
                    )
                };
                info.as_lapack_result()?;
                Ok(unsafe { self.ipiv.slice_assume_init_ref() })
            }

            fn eval(mut self, uplo: UPLO, a: &mut [Self::Elem]) -> Result<Pivot> {
                let _ref = self.calc(uplo, a)?;
                Ok(unsafe { self.ipiv.assume_init() })
            }
        }
    };
}
impl_bk_work!(c64, lapack_sys::zhetrf_);
impl_bk_work!(c32, lapack_sys::chetrf_);
impl_bk_work!(f64, lapack_sys::dsytrf_);
impl_bk_work!(f32, lapack_sys::ssytrf_);

pub struct InvhWork<T: Scalar> {
    pub layout: MatrixLayout,
    pub work: Vec<MaybeUninit<T>>,
}

pub trait InvhWorkImpl: Sized {
    type Elem;
    fn new(layout: MatrixLayout) -> Result<Self>;
    fn calc(&mut self, uplo: UPLO, a: &mut [Self::Elem], ipiv: &Pivot) -> Result<()>;
}

macro_rules! impl_invh_work {
    ($s:ty, $tri:path) => {
        impl InvhWorkImpl for InvhWork<$s> {
            type Elem = $s;

            fn new(layout: MatrixLayout) -> Result<Self> {
                let (n, _) = layout.size();
                let work = vec_uninit(n as usize);
                Ok(InvhWork { layout, work })
            }

            fn calc(&mut self, uplo: UPLO, a: &mut [Self::Elem], ipiv: &Pivot) -> Result<()> {
                let (n, _) = self.layout.size();
                let mut info = 0;
                unsafe {
                    $tri(
                        uplo.as_ptr(),
                        &n,
                        AsPtr::as_mut_ptr(a),
                        &self.layout.lda(),
                        ipiv.as_ptr(),
                        AsPtr::as_mut_ptr(&mut self.work),
                        &mut info,
                    )
                };
                info.as_lapack_result()?;
                Ok(())
            }
        }
    };
}
impl_invh_work!(c64, lapack_sys::zhetri_);
impl_invh_work!(c32, lapack_sys::chetri_);
impl_invh_work!(f64, lapack_sys::dsytri_);
impl_invh_work!(f32, lapack_sys::ssytri_);

pub trait SolvehImpl: Scalar {
    fn solveh(l: MatrixLayout, uplo: UPLO, a: &[Self], ipiv: &Pivot, b: &mut [Self]) -> Result<()>;
}

macro_rules! impl_solveh_ {
    ($s:ty, $trs:path) => {
        impl SolvehImpl for $s {
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
}

impl_solveh_!(c64, lapack_sys::zhetrs_);
impl_solveh_!(c32, lapack_sys::chetrs_);
impl_solveh_!(f64, lapack_sys::dsytrs_);
impl_solveh_!(f32, lapack_sys::ssytrs_);

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
                let mut ipiv = vec_uninit(n as usize);
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
                let mut work: Vec<MaybeUninit<Self>> = vec_uninit(lwork);
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
                let mut work: Vec<MaybeUninit<Self>> = vec_uninit(n as usize);
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
