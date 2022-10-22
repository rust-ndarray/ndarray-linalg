//! QR decomposition

use crate::{error::*, layout::MatrixLayout, *};
use cauchy::*;
use num_traits::{ToPrimitive, Zero};

pub struct HouseholderWork<T: Scalar> {
    pub m: i32,
    pub n: i32,
    pub layout: MatrixLayout,
    pub tau: Vec<MaybeUninit<T>>,
    pub work: Vec<MaybeUninit<T>>,
}

pub trait HouseholderWorkImpl: Sized {
    type Elem: Scalar;
    fn new(l: MatrixLayout) -> Result<Self>;
    fn calc(&mut self, a: &mut [Self::Elem]) -> Result<&[Self::Elem]>;
    fn eval(self, a: &mut [Self::Elem]) -> Result<Vec<Self::Elem>>;
}

macro_rules! impl_householder_work {
    ($s:ty, $qrf:path, $lqf: path) => {
        impl HouseholderWorkImpl for HouseholderWork<$s> {
            type Elem = $s;

            fn new(layout: MatrixLayout) -> Result<Self> {
                let m = layout.lda();
                let n = layout.len();
                let k = m.min(n);
                let mut tau = vec_uninit(k as usize);
                let mut info = 0;
                let mut work_size = [Self::Elem::zero()];
                match layout {
                    MatrixLayout::F { .. } => unsafe {
                        $qrf(
                            &m,
                            &n,
                            std::ptr::null_mut(),
                            &m,
                            AsPtr::as_mut_ptr(&mut tau),
                            AsPtr::as_mut_ptr(&mut work_size),
                            &(-1),
                            &mut info,
                        )
                    },
                    MatrixLayout::C { .. } => unsafe {
                        $lqf(
                            &m,
                            &n,
                            std::ptr::null_mut(),
                            &m,
                            AsPtr::as_mut_ptr(&mut tau),
                            AsPtr::as_mut_ptr(&mut work_size),
                            &(-1),
                            &mut info,
                        )
                    },
                }
                info.as_lapack_result()?;
                let lwork = work_size[0].to_usize().unwrap();
                let work = vec_uninit(lwork);
                Ok(HouseholderWork {
                    n,
                    m,
                    layout,
                    tau,
                    work,
                })
            }

            fn calc(&mut self, a: &mut [Self::Elem]) -> Result<&[Self::Elem]> {
                let lwork = self.work.len().to_i32().unwrap();
                let mut info = 0;
                match self.layout {
                    MatrixLayout::F { .. } => unsafe {
                        $qrf(
                            &self.m,
                            &self.n,
                            AsPtr::as_mut_ptr(a),
                            &self.m,
                            AsPtr::as_mut_ptr(&mut self.tau),
                            AsPtr::as_mut_ptr(&mut self.work),
                            &lwork,
                            &mut info,
                        );
                    },
                    MatrixLayout::C { .. } => unsafe {
                        $lqf(
                            &self.m,
                            &self.n,
                            AsPtr::as_mut_ptr(a),
                            &self.m,
                            AsPtr::as_mut_ptr(&mut self.tau),
                            AsPtr::as_mut_ptr(&mut self.work),
                            &lwork,
                            &mut info,
                        );
                    },
                }
                info.as_lapack_result()?;
                Ok(unsafe { self.tau.slice_assume_init_ref() })
            }

            fn eval(mut self, a: &mut [Self::Elem]) -> Result<Vec<Self::Elem>> {
                let _eig = self.calc(a)?;
                Ok(unsafe { self.tau.assume_init() })
            }
        }
    };
}
impl_householder_work!(c64, lapack_sys::zgeqrf_, lapack_sys::zgelqf_);
impl_householder_work!(c32, lapack_sys::cgeqrf_, lapack_sys::cgelqf_);
impl_householder_work!(f64, lapack_sys::dgeqrf_, lapack_sys::dgelqf_);
impl_householder_work!(f32, lapack_sys::sgeqrf_, lapack_sys::sgelqf_);

pub struct QWork<T: Scalar> {
    pub layout: MatrixLayout,
    pub work: Vec<MaybeUninit<T>>,
}

pub trait QWorkImpl: Sized {
    type Elem: Scalar;
    fn new(layout: MatrixLayout) -> Result<Self>;
    fn calc(&mut self, a: &mut [Self::Elem], tau: &[Self::Elem]) -> Result<()>;
}

macro_rules! impl_q_work {
    ($s:ty, $gqr:path, $glq:path) => {
        impl QWorkImpl for QWork<$s> {
            type Elem = $s;

            fn new(layout: MatrixLayout) -> Result<Self> {
                let m = layout.lda();
                let n = layout.len();
                let k = m.min(n);
                let mut info = 0;
                let mut work_size = [Self::Elem::zero()];
                match layout {
                    MatrixLayout::F { .. } => unsafe {
                        $gqr(
                            &m,
                            &k,
                            &k,
                            std::ptr::null_mut(),
                            &m,
                            std::ptr::null_mut(),
                            AsPtr::as_mut_ptr(&mut work_size),
                            &(-1),
                            &mut info,
                        )
                    },
                    MatrixLayout::C { .. } => unsafe {
                        $glq(
                            &k,
                            &n,
                            &k,
                            std::ptr::null_mut(),
                            &m,
                            std::ptr::null_mut(),
                            AsPtr::as_mut_ptr(&mut work_size),
                            &(-1),
                            &mut info,
                        )
                    },
                }
                let lwork = work_size[0].to_usize().unwrap();
                let work = vec_uninit(lwork);
                Ok(QWork { layout, work })
            }

            fn calc(&mut self, a: &mut [Self::Elem], tau: &[Self::Elem]) -> Result<()> {
                let m = self.layout.lda();
                let n = self.layout.len();
                let k = m.min(n);
                let lwork = self.work.len().to_i32().unwrap();
                let mut info = 0;
                match self.layout {
                    MatrixLayout::F { .. } => unsafe {
                        $gqr(
                            &m,
                            &k,
                            &k,
                            AsPtr::as_mut_ptr(a),
                            &m,
                            AsPtr::as_ptr(&tau),
                            AsPtr::as_mut_ptr(&mut self.work),
                            &lwork,
                            &mut info,
                        )
                    },
                    MatrixLayout::C { .. } => unsafe {
                        $glq(
                            &k,
                            &n,
                            &k,
                            AsPtr::as_mut_ptr(a),
                            &m,
                            AsPtr::as_ptr(&tau),
                            AsPtr::as_mut_ptr(&mut self.work),
                            &lwork,
                            &mut info,
                        )
                    },
                }
                info.as_lapack_result()?;
                Ok(())
            }
        }
    };
}

impl_q_work!(c64, lapack_sys::zungqr_, lapack_sys::zunglq_);
impl_q_work!(c32, lapack_sys::cungqr_, lapack_sys::cunglq_);
impl_q_work!(f64, lapack_sys::dorgqr_, lapack_sys::dorglq_);
impl_q_work!(f32, lapack_sys::sorgqr_, lapack_sys::sorglq_);
