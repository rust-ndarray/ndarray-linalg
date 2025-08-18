//! Compute singular value decomposition with divide-and-conquer algorithm
//!
//! LAPACK correspondance
//! ----------------------
//!
//! | f32    | f64    | c32    | c64    |
//! |:-------|:-------|:-------|:-------|
//! | sgesdd | dgesdd | cgesdd | zgesdd |
//!

use crate::{error::*, *};
use cauchy::*;
use num_traits::{ToPrimitive, Zero};

pub struct SvdDcWork<T: Scalar> {
    pub jobz: JobSvd,
    pub layout: MatrixLayout,
    pub s: Vec<MaybeUninit<T::Real>>,
    pub u: Option<Vec<MaybeUninit<T>>>,
    pub vt: Option<Vec<MaybeUninit<T>>>,
    pub work: Vec<MaybeUninit<T>>,
    pub iwork: Vec<MaybeUninit<i32>>,
    pub rwork: Option<Vec<MaybeUninit<T::Real>>>,
}

pub trait SvdDcWorkImpl: Sized {
    type Elem: Scalar;
    fn new(layout: MatrixLayout, jobz: JobSvd) -> Result<Self>;
    fn calc(&mut self, a: &mut [Self::Elem]) -> Result<SvdRef<Self::Elem>>;
    fn eval(self, a: &mut [Self::Elem]) -> Result<SvdOwned<Self::Elem>>;
}

macro_rules! impl_svd_dc_work_c {
    ($s:ty, $sdd:path) => {
        impl SvdDcWorkImpl for SvdDcWork<$s> {
            type Elem = $s;

            fn new(layout: MatrixLayout, jobz: JobSvd) -> Result<Self> {
                let m = layout.lda();
                let n = layout.len();
                let k = m.min(n);
                let (u_col, vt_row) = match jobz {
                    JobSvd::All | JobSvd::None => (m, n),
                    JobSvd::Some => (k, k),
                };

                let mut s = vec_uninit(k as usize);
                let (mut u, mut vt) = match jobz {
                    JobSvd::All => (
                        Some(vec_uninit((m * m) as usize)),
                        Some(vec_uninit((n * n) as usize)),
                    ),
                    JobSvd::Some => (
                        Some(vec_uninit((m * u_col) as usize)),
                        Some(vec_uninit((n * vt_row) as usize)),
                    ),
                    JobSvd::None => (None, None),
                };
                let mut iwork = vec_uninit(8 * k as usize);

                let mx = n.max(m) as usize;
                let mn = n.min(m) as usize;
                let lrwork = match jobz {
                    JobSvd::None => 7 * mn,
                    _ => std::cmp::max(5 * mn * mn + 5 * mn, 2 * mx * mn + 2 * mn * mn + mn),
                };
                let mut rwork = vec_uninit(lrwork);

                let mut info = 0;
                let mut work_size = [Self::Elem::zero()];
                unsafe {
                    $sdd(
                        jobz.as_ptr(),
                        &m,
                        &n,
                        std::ptr::null_mut(),
                        &m,
                        AsPtr::as_mut_ptr(&mut s),
                        AsPtr::as_mut_ptr(u.as_mut().map(|x| x.as_mut_slice()).unwrap_or(&mut [])),
                        &m,
                        AsPtr::as_mut_ptr(vt.as_mut().map(|x| x.as_mut_slice()).unwrap_or(&mut [])),
                        &vt_row,
                        AsPtr::as_mut_ptr(&mut work_size),
                        &(-1),
                        AsPtr::as_mut_ptr(&mut rwork),
                        AsPtr::as_mut_ptr(&mut iwork),
                        &mut info,
                    );
                }
                info.as_lapack_result()?;
                let lwork = work_size[0].to_usize().unwrap();
                let work = vec_uninit(lwork);
                Ok(SvdDcWork {
                    layout,
                    jobz,
                    iwork,
                    work,
                    rwork: Some(rwork),
                    u,
                    vt,
                    s,
                })
            }

            fn calc(&mut self, a: &mut [Self::Elem]) -> Result<SvdRef<Self::Elem>> {
                let m = self.layout.lda();
                let n = self.layout.len();
                let k = m.min(n);
                let (_, vt_row) = match self.jobz {
                    JobSvd::All | JobSvd::None => (m, n),
                    JobSvd::Some => (k, k),
                };
                let lwork = self.work.len().to_i32().unwrap();

                let mut info = 0;
                unsafe {
                    $sdd(
                        self.jobz.as_ptr(),
                        &m,
                        &n,
                        AsPtr::as_mut_ptr(a),
                        &m,
                        AsPtr::as_mut_ptr(&mut self.s),
                        AsPtr::as_mut_ptr(
                            self.u.as_mut().map(|x| x.as_mut_slice()).unwrap_or(&mut []),
                        ),
                        &m,
                        AsPtr::as_mut_ptr(
                            self.vt
                                .as_mut()
                                .map(|x| x.as_mut_slice())
                                .unwrap_or(&mut []),
                        ),
                        &vt_row,
                        AsPtr::as_mut_ptr(&mut self.work),
                        &lwork,
                        AsPtr::as_mut_ptr(self.rwork.as_mut().unwrap()),
                        AsPtr::as_mut_ptr(&mut self.iwork),
                        &mut info,
                    );
                }
                info.as_lapack_result()?;

                let s = unsafe { self.s.slice_assume_init_ref() };
                let u = self
                    .u
                    .as_ref()
                    .map(|v| unsafe { v.slice_assume_init_ref() });
                let vt = self
                    .vt
                    .as_ref()
                    .map(|v| unsafe { v.slice_assume_init_ref() });

                Ok(match self.layout {
                    MatrixLayout::F { .. } => SvdRef { s, u, vt },
                    MatrixLayout::C { .. } => SvdRef { s, u: vt, vt: u },
                })
            }

            fn eval(mut self, a: &mut [Self::Elem]) -> Result<SvdOwned<Self::Elem>> {
                let _ref = self.calc(a)?;
                let s = unsafe { self.s.assume_init() };
                let u = self.u.map(|v| unsafe { v.assume_init() });
                let vt = self.vt.map(|v| unsafe { v.assume_init() });
                Ok(match self.layout {
                    MatrixLayout::F { .. } => SvdOwned { s, u, vt },
                    MatrixLayout::C { .. } => SvdOwned { s, u: vt, vt: u },
                })
            }
        }
    };
}
impl_svd_dc_work_c!(c64, lapack_sys::zgesdd_);
impl_svd_dc_work_c!(c32, lapack_sys::cgesdd_);

macro_rules! impl_svd_dc_work_r {
    ($s:ty, $sdd:path) => {
        impl SvdDcWorkImpl for SvdDcWork<$s> {
            type Elem = $s;

            fn new(layout: MatrixLayout, jobz: JobSvd) -> Result<Self> {
                let m = layout.lda();
                let n = layout.len();
                let k = m.min(n);
                let (u_col, vt_row) = match jobz {
                    JobSvd::All | JobSvd::None => (m, n),
                    JobSvd::Some => (k, k),
                };

                let mut s = vec_uninit(k as usize);
                let (mut u, mut vt) = match jobz {
                    JobSvd::All => (
                        Some(vec_uninit((m * m) as usize)),
                        Some(vec_uninit((n * n) as usize)),
                    ),
                    JobSvd::Some => (
                        Some(vec_uninit((m * u_col) as usize)),
                        Some(vec_uninit((n * vt_row) as usize)),
                    ),
                    JobSvd::None => (None, None),
                };
                let mut iwork = vec_uninit(8 * k as usize);

                let mut info = 0;
                let mut work_size = [Self::Elem::zero()];
                unsafe {
                    $sdd(
                        jobz.as_ptr(),
                        &m,
                        &n,
                        std::ptr::null_mut(),
                        &m,
                        AsPtr::as_mut_ptr(&mut s),
                        AsPtr::as_mut_ptr(u.as_mut().map(|x| x.as_mut_slice()).unwrap_or(&mut [])),
                        &m,
                        AsPtr::as_mut_ptr(vt.as_mut().map(|x| x.as_mut_slice()).unwrap_or(&mut [])),
                        &vt_row,
                        AsPtr::as_mut_ptr(&mut work_size),
                        &(-1),
                        AsPtr::as_mut_ptr(&mut iwork),
                        &mut info,
                    );
                }
                info.as_lapack_result()?;
                let lwork = work_size[0].to_usize().unwrap();
                let work = vec_uninit(lwork);
                Ok(SvdDcWork {
                    layout,
                    jobz,
                    iwork,
                    work,
                    rwork: None,
                    u,
                    vt,
                    s,
                })
            }

            fn calc(&mut self, a: &mut [Self::Elem]) -> Result<SvdRef<Self::Elem>> {
                let m = self.layout.lda();
                let n = self.layout.len();
                let k = m.min(n);
                let (_, vt_row) = match self.jobz {
                    JobSvd::All | JobSvd::None => (m, n),
                    JobSvd::Some => (k, k),
                };
                let lwork = self.work.len().to_i32().unwrap();

                let mut info = 0;
                unsafe {
                    $sdd(
                        self.jobz.as_ptr(),
                        &m,
                        &n,
                        AsPtr::as_mut_ptr(a),
                        &m,
                        AsPtr::as_mut_ptr(&mut self.s),
                        AsPtr::as_mut_ptr(
                            self.u.as_mut().map(|x| x.as_mut_slice()).unwrap_or(&mut []),
                        ),
                        &m,
                        AsPtr::as_mut_ptr(
                            self.vt
                                .as_mut()
                                .map(|x| x.as_mut_slice())
                                .unwrap_or(&mut []),
                        ),
                        &vt_row,
                        AsPtr::as_mut_ptr(&mut self.work),
                        &lwork,
                        AsPtr::as_mut_ptr(&mut self.iwork),
                        &mut info,
                    );
                }
                info.as_lapack_result()?;

                let s = unsafe { self.s.slice_assume_init_ref() };
                let u = self
                    .u
                    .as_ref()
                    .map(|v| unsafe { v.slice_assume_init_ref() });
                let vt = self
                    .vt
                    .as_ref()
                    .map(|v| unsafe { v.slice_assume_init_ref() });

                Ok(match self.layout {
                    MatrixLayout::F { .. } => SvdRef { s, u, vt },
                    MatrixLayout::C { .. } => SvdRef { s, u: vt, vt: u },
                })
            }

            fn eval(mut self, a: &mut [Self::Elem]) -> Result<SvdOwned<Self::Elem>> {
                let _ref = self.calc(a)?;
                let s = unsafe { self.s.assume_init() };
                let u = self.u.map(|v| unsafe { v.assume_init() });
                let vt = self.vt.map(|v| unsafe { v.assume_init() });
                Ok(match self.layout {
                    MatrixLayout::F { .. } => SvdOwned { s, u, vt },
                    MatrixLayout::C { .. } => SvdOwned { s, u: vt, vt: u },
                })
            }
        }
    };
}
impl_svd_dc_work_r!(f64, lapack_sys::dgesdd_);
impl_svd_dc_work_r!(f32, lapack_sys::sgesdd_);
