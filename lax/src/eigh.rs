//! Eigenvalue problem for symmetric/Hermite matricies
//!
//! LAPACK correspondance
//! ----------------------
//!
//! | f32   | f64   | c32   | c64   |
//! |:------|:------|:------|:------|
//! | ssyev | dsyev | cheev | zheev |

use super::*;
use crate::{error::*, layout::MatrixLayout};
use cauchy::*;
use num_traits::{ToPrimitive, Zero};

pub struct EighWork<T: Scalar> {
    pub n: i32,
    pub jobz: JobEv,
    pub eigs: Vec<MaybeUninit<T::Real>>,
    pub work: Vec<MaybeUninit<T>>,
    pub rwork: Option<Vec<MaybeUninit<T::Real>>>,
}

pub trait EighWorkImpl: Sized {
    type Elem: Scalar;
    fn new(calc_eigenvectors: bool, layout: MatrixLayout) -> Result<Self>;
    fn calc(&mut self, uplo: UPLO, a: &mut [Self::Elem])
        -> Result<&[<Self::Elem as Scalar>::Real]>;
    fn eval(self, uplo: UPLO, a: &mut [Self::Elem]) -> Result<Vec<<Self::Elem as Scalar>::Real>>;
}

macro_rules! impl_eigh_work_c {
    ($c:ty, $ev:path) => {
        impl EighWorkImpl for EighWork<$c> {
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
                    $ev(
                        jobz.as_ptr(),
                        UPLO::Upper.as_ptr(), // dummy, working memory is not affected by UPLO
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
                Ok(EighWork {
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
            ) -> Result<&[<Self::Elem as Scalar>::Real]> {
                let lwork = self.work.len().to_i32().unwrap();
                let mut info = 0;
                unsafe {
                    $ev(
                        self.jobz.as_ptr(),
                        uplo.as_ptr(),
                        &self.n,
                        AsPtr::as_mut_ptr(a),
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
            ) -> Result<Vec<<Self::Elem as Scalar>::Real>> {
                let _eig = self.calc(uplo, a)?;
                Ok(unsafe { self.eigs.assume_init() })
            }
        }
    };
}
impl_eigh_work_c!(c64, lapack_sys::zheev_);
impl_eigh_work_c!(c32, lapack_sys::cheev_);

macro_rules! impl_eigh_work_r {
    ($f:ty, $ev:path) => {
        impl EighWorkImpl for EighWork<$f> {
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
                    $ev(
                        jobz.as_ptr(),
                        UPLO::Upper.as_ptr(), // dummy, working memory is not affected by UPLO
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
                Ok(EighWork {
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
            ) -> Result<&[<Self::Elem as Scalar>::Real]> {
                let lwork = self.work.len().to_i32().unwrap();
                let mut info = 0;
                unsafe {
                    $ev(
                        self.jobz.as_ptr(),
                        uplo.as_ptr(),
                        &self.n,
                        AsPtr::as_mut_ptr(a),
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
            ) -> Result<Vec<<Self::Elem as Scalar>::Real>> {
                let _eig = self.calc(uplo, a)?;
                Ok(unsafe { self.eigs.assume_init() })
            }
        }
    };
}
impl_eigh_work_r!(f64, lapack_sys::dsyev_);
impl_eigh_work_r!(f32, lapack_sys::ssyev_);
