use crate::{error::*, layout::MatrixLayout, *};
use cauchy::*;
use num_traits::{ToPrimitive, Zero};

pub struct BkWork<T: Scalar> {
    pub layout: MatrixLayout,
    pub work: Vec<MaybeUninit<T>>,
    pub ipiv: Vec<MaybeUninit<i32>>,
}

/// Factorize symmetric/Hermitian matrix using Bunch-Kaufman diagonal pivoting method
///
/// LAPACK correspondance
/// ----------------------
///
/// | f32    | f64    | c32    | c64    |
/// |:-------|:-------|:-------|:-------|
/// | ssytrf | dsytrf | chetrf | zhetrf |
///
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
                if lwork == 0 {
                    return Ok(&[]);
                }
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

/// Compute inverse matrix of symmetric/Hermitian matrix
///
/// LAPACK correspondance
/// ----------------------
///
/// | f32    | f64    | c32    | c64    |
/// |:-------|:-------|:-------|:-------|
/// | ssytri | dsytri | chetri | zhetri |
///
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

/// Solve symmetric/Hermitian linear equation
///
/// LAPACK correspondance
/// ----------------------
///
/// | f32    | f64    | c32    | c64    |
/// |:-------|:-------|:-------|:-------|
/// | ssytrs | dsytrs | chetrs | zhetrs |
///
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
