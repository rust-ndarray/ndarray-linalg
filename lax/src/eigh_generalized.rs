//! Compute generalized right eigenvalue and eigenvectors
//!
//! LAPACK correspondance
//! ----------------------
//!
//! | f32   | f64   | c32   | c64   |
//! |:------|:------|:------|:------|
//! | ssygv | dsygv | chegv | zhegv |
//!

use super::*;
use crate::{error::*, layout::MatrixLayout};
use cauchy::*;
use num_traits::{ToPrimitive, Zero};

pub struct EighGeneralizedWork<T: Scalar> {
    pub n: i32,
    pub jobz: JobEv,
    pub eigs: Vec<MaybeUninit<T::Real>>,
    pub work: Vec<MaybeUninit<T>>,
    pub rwork: Option<Vec<MaybeUninit<T::Real>>>,
}

pub trait EighGeneralizedWorkImpl: Sized {
    type Elem: Scalar;
    fn new(calc_eigenvectors: bool, layout: MatrixLayout) -> Result<Self>;
    fn calc(
        &mut self,
        uplo: UPLO,
        a: &mut [Self::Elem],
        b: &mut [Self::Elem],
    ) -> Result<&[<Self::Elem as Scalar>::Real]>;
    fn eval(
        self,
        uplo: UPLO,
        a: &mut [Self::Elem],
        b: &mut [Self::Elem],
    ) -> Result<Vec<<Self::Elem as Scalar>::Real>>;
}

macro_rules! impl_eigh_generalized_work_c {
    ($c:ty, $gv:path) => {
        impl EighGeneralizedWorkImpl for EighGeneralizedWork<$c> {
            type Elem = $c;

            fn new(calc_eigenvectors: bool, layout: MatrixLayout) -> Result<Self> {
                assert_eq!(layout.len(), layout.lda());
                let n = layout.len();
                let jobz = if calc_eigenvectors {
                    JobEv::All
                } else {
                    JobEv::None
                };
                let mut eigs = vec_uninit(n as usize);
                let mut rwork = vec_uninit(3 * n as usize - 2 as usize);
                let mut info = 0;
                let mut work_size = [Self::Elem::zero()];
                unsafe {
                    $gv(
                        &1, // ITYPE A*x = (lambda)*B*x
                        jobz.as_ptr(),
                        UPLO::Upper.as_ptr(), // dummy, working memory is not affected by UPLO
                        &n,
                        std::ptr::null_mut(),
                        &n,
                        std::ptr::null_mut(),
                        &n,
                        AsPtr::as_mut_ptr(&mut eigs),
                        AsPtr::as_mut_ptr(&mut work_size),
                        &(-1),
                        AsPtr::as_mut_ptr(&mut rwork),
                        &mut info,
                    );
                }
                info.as_lapack_result()?;
                let lwork = work_size[0].to_usize().unwrap();
                let work = vec_uninit(lwork);
                Ok(EighGeneralizedWork {
                    n,
                    eigs,
                    jobz,
                    work,
                    rwork: Some(rwork),
                })
            }

            fn calc(
                &mut self,
                uplo: UPLO,
                a: &mut [Self::Elem],
                b: &mut [Self::Elem],
            ) -> Result<&[<Self::Elem as Scalar>::Real]> {
                let lwork = self.work.len().to_i32().unwrap();
                let mut info = 0;
                unsafe {
                    $gv(
                        &1, // ITYPE A*x = (lambda)*B*x
                        self.jobz.as_ptr(),
                        uplo.as_ptr(),
                        &self.n,
                        AsPtr::as_mut_ptr(a),
                        &self.n,
                        AsPtr::as_mut_ptr(b),
                        &self.n,
                        AsPtr::as_mut_ptr(&mut self.eigs),
                        AsPtr::as_mut_ptr(&mut self.work),
                        &lwork,
                        AsPtr::as_mut_ptr(self.rwork.as_mut().unwrap()),
                        &mut info,
                    );
                }
                info.as_lapack_result()?;
                Ok(unsafe { self.eigs.slice_assume_init_ref() })
            }

            fn eval(
                mut self,
                uplo: UPLO,
                a: &mut [Self::Elem],
                b: &mut [Self::Elem],
            ) -> Result<Vec<<Self::Elem as Scalar>::Real>> {
                let _eig = self.calc(uplo, a, b)?;
                Ok(unsafe { self.eigs.assume_init() })
            }
        }
    };
}
impl_eigh_generalized_work_c!(c64, lapack_sys::zhegv_);
impl_eigh_generalized_work_c!(c32, lapack_sys::chegv_);

macro_rules! impl_eigh_generalized_work_r {
    ($f:ty, $gv:path) => {
        impl EighGeneralizedWorkImpl for EighGeneralizedWork<$f> {
            type Elem = $f;

            fn new(calc_eigenvectors: bool, layout: MatrixLayout) -> Result<Self> {
                assert_eq!(layout.len(), layout.lda());
                let n = layout.len();
                let jobz = if calc_eigenvectors {
                    JobEv::All
                } else {
                    JobEv::None
                };
                let mut eigs = vec_uninit(n as usize);
                let mut info = 0;
                let mut work_size = [Self::Elem::zero()];
                unsafe {
                    $gv(
                        &1, // ITYPE A*x = (lambda)*B*x
                        jobz.as_ptr(),
                        UPLO::Upper.as_ptr(), // dummy, working memory is not affected by UPLO
                        &n,
                        std::ptr::null_mut(),
                        &n,
                        std::ptr::null_mut(),
                        &n,
                        AsPtr::as_mut_ptr(&mut eigs),
                        AsPtr::as_mut_ptr(&mut work_size),
                        &(-1),
                        &mut info,
                    );
                }
                info.as_lapack_result()?;
                let lwork = work_size[0].to_usize().unwrap();
                let work = vec_uninit(lwork);
                Ok(EighGeneralizedWork {
                    n,
                    eigs,
                    jobz,
                    work,
                    rwork: None,
                })
            }

            fn calc(
                &mut self,
                uplo: UPLO,
                a: &mut [Self::Elem],
                b: &mut [Self::Elem],
            ) -> Result<&[<Self::Elem as Scalar>::Real]> {
                let lwork = self.work.len().to_i32().unwrap();
                let mut info = 0;
                unsafe {
                    $gv(
                        &1, // ITYPE A*x = (lambda)*B*x
                        self.jobz.as_ptr(),
                        uplo.as_ptr(),
                        &self.n,
                        AsPtr::as_mut_ptr(a),
                        &self.n,
                        AsPtr::as_mut_ptr(b),
                        &self.n,
                        AsPtr::as_mut_ptr(&mut self.eigs),
                        AsPtr::as_mut_ptr(&mut self.work),
                        &lwork,
                        &mut info,
                    );
                }
                info.as_lapack_result()?;
                Ok(unsafe { self.eigs.slice_assume_init_ref() })
            }

            fn eval(
                mut self,
                uplo: UPLO,
                a: &mut [Self::Elem],
                b: &mut [Self::Elem],
            ) -> Result<Vec<<Self::Elem as Scalar>::Real>> {
                let _eig = self.calc(uplo, a, b)?;
                Ok(unsafe { self.eigs.assume_init() })
            }
        }
    };
}
impl_eigh_generalized_work_r!(f64, lapack_sys::dsygv_);
impl_eigh_generalized_work_r!(f32, lapack_sys::ssygv_);

#[cfg_attr(doc, katexit::katexit)]
/// Eigenvalue problem for symmetric/hermite matrix
pub trait EighGeneralized_: Scalar {
    /// Compute generalized right eigenvalue and eigenvectors $Ax = \lambda B x$
    ///
    /// LAPACK correspondance
    /// ----------------------
    ///
    /// | f32   | f64   | c32   | c64   |
    /// |:------|:------|:------|:------|
    /// | ssygv | dsygv | chegv | zhegv |
    ///
    fn eigh_generalized(
        calc_eigenvec: bool,
        layout: MatrixLayout,
        uplo: UPLO,
        a: &mut [Self],
        b: &mut [Self],
    ) -> Result<Vec<Self::Real>>;
}

macro_rules! impl_eigh {
    (@real, $scalar:ty, $evg:path) => {
        impl_eigh!(@body, $scalar, $evg, );
    };
    (@complex, $scalar:ty, $evg:path) => {
        impl_eigh!(@body, $scalar, $evg, rwork);
    };
    (@body, $scalar:ty, $evg:path, $($rwork_ident:ident),*) => {
        impl EighGeneralized_ for $scalar {
            fn eigh_generalized(
                calc_v: bool,
                layout: MatrixLayout,
                uplo: UPLO,
                a: &mut [Self],
                b: &mut [Self],
            ) -> Result<Vec<Self::Real>> {
                assert_eq!(layout.len(), layout.lda());
                let n = layout.len();
                let jobz = if calc_v { JobEv::All } else { JobEv::None };
                let mut eigs: Vec<MaybeUninit<Self::Real>> = vec_uninit(n as usize);

                $(
                let mut $rwork_ident: Vec<MaybeUninit<Self::Real>> = vec_uninit(3 * n as usize - 2);
                )*

                // calc work size
                let mut info = 0;
                let mut work_size = [Self::zero()];
                unsafe {
                    $evg(
                        &1, // ITYPE A*x = (lambda)*B*x
                        jobz.as_ptr(),
                        uplo.as_ptr(),
                        &n,
                        AsPtr::as_mut_ptr(a),
                        &n,
                        AsPtr::as_mut_ptr(b),
                        &n,
                        AsPtr::as_mut_ptr(&mut eigs),
                        AsPtr::as_mut_ptr(&mut work_size),
                        &(-1),
                        $(AsPtr::as_mut_ptr(&mut $rwork_ident),)*
                        &mut info,
                    );
                }
                info.as_lapack_result()?;

                // actual evg
                let lwork = work_size[0].to_usize().unwrap();
                let mut work: Vec<MaybeUninit<Self>> = vec_uninit(lwork);
                let lwork = lwork as i32;
                unsafe {
                    $evg(
                        &1, // ITYPE A*x = (lambda)*B*x
                        jobz.as_ptr(),
                        uplo.as_ptr(),
                        &n,
                        AsPtr::as_mut_ptr(a),
                        &n,
                        AsPtr::as_mut_ptr(b),
                        &n,
                        AsPtr::as_mut_ptr(&mut eigs),
                        AsPtr::as_mut_ptr(&mut work),
                        &lwork,
                        $(AsPtr::as_mut_ptr(&mut $rwork_ident),)*
                        &mut info,
                    );
                }
                info.as_lapack_result()?;
                let eigs = unsafe { eigs.assume_init() };
                Ok(eigs)
            }
        }
    };
} // impl_eigh!

impl_eigh!(@real, f64, lapack_sys::dsygv_);
impl_eigh!(@real, f32, lapack_sys::ssygv_);
impl_eigh!(@complex, c64, lapack_sys::zhegv_);
impl_eigh!(@complex, c32, lapack_sys::chegv_);
