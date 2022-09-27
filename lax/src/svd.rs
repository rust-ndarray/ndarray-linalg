//! Singular-value decomposition

use super::{error::*, layout::*, *};
use cauchy::*;
use num_traits::{ToPrimitive, Zero};

/// Result of SVD
pub struct SVDOutput<A: Scalar> {
    /// diagonal values
    pub s: Vec<A::Real>,
    /// Unitary matrix for destination space
    pub u: Option<Vec<A>>,
    /// Unitary matrix for departure space
    pub vt: Option<Vec<A>>,
}

#[cfg_attr(doc, katexit::katexit)]
/// Singular value decomposition
pub trait SVD_: Scalar {
    /// Compute singular value decomposition $A = U \Sigma V^T$
    ///
    /// LAPACK correspondance
    /// ----------------------
    ///
    /// | f32    | f64    | c32    | c64    |
    /// |:-------|:-------|:-------|:-------|
    /// | sgesvd | dgesvd | cgesvd | zgesvd |
    ///
    fn svd(l: MatrixLayout, calc_u: bool, calc_vt: bool, a: &mut [Self])
        -> Result<SVDOutput<Self>>;
}

pub struct SvdWork<T: Scalar> {
    pub ju: JobSvd,
    pub jvt: JobSvd,
    pub layout: MatrixLayout,
    pub s: Vec<MaybeUninit<T::Real>>,
    pub u: Option<Vec<MaybeUninit<T>>>,
    pub vt: Option<Vec<MaybeUninit<T>>>,
    pub work: Vec<MaybeUninit<T>>,
    pub rwork: Option<Vec<MaybeUninit<T::Real>>>,
}

#[derive(Debug, Clone)]
pub struct SvdRef<'work, T: Scalar> {
    pub s: &'work [T::Real],
    pub u: Option<&'work [T]>,
    pub vt: Option<&'work [T]>,
}

#[derive(Debug, Clone)]
pub struct SvdOwned<T: Scalar> {
    pub s: Vec<T::Real>,
    pub u: Option<Vec<T>>,
    pub vt: Option<Vec<T>>,
}

pub trait SvdWorkImpl: Sized {
    type Elem: Scalar;
    fn new(layout: MatrixLayout, calc_u: bool, calc_vt: bool) -> Result<Self>;
    fn calc(&mut self, a: &mut [Self::Elem]) -> Result<SvdRef<Self::Elem>>;
    fn eval(self, a: &mut [Self::Elem]) -> Result<SvdOwned<Self::Elem>>;
}

macro_rules! impl_svd_work_c {
    ($s:ty, $svd:path) => {
        impl SvdWorkImpl for SvdWork<$s> {
            type Elem = $s;

            fn new(layout: MatrixLayout, calc_u: bool, calc_vt: bool) -> Result<Self> {
                let ju = match layout {
                    MatrixLayout::F { .. } => JobSvd::from_bool(calc_u),
                    MatrixLayout::C { .. } => JobSvd::from_bool(calc_vt),
                };
                let jvt = match layout {
                    MatrixLayout::F { .. } => JobSvd::from_bool(calc_vt),
                    MatrixLayout::C { .. } => JobSvd::from_bool(calc_u),
                };

                let m = layout.lda();
                let mut u = match ju {
                    JobSvd::All => Some(vec_uninit((m * m) as usize)),
                    JobSvd::None => None,
                    _ => unimplemented!("SVD with partial vector output is not supported yet"),
                };

                let n = layout.len();
                let mut vt = match jvt {
                    JobSvd::All => Some(vec_uninit((n * n) as usize)),
                    JobSvd::None => None,
                    _ => unimplemented!("SVD with partial vector output is not supported yet"),
                };

                let k = std::cmp::min(m, n);
                let mut s = vec_uninit(k as usize);
                let mut rwork = vec_uninit(5 * k as usize);

                // eval work size
                let mut info = 0;
                let mut work_size = [Self::Elem::zero()];
                unsafe {
                    $svd(
                        ju.as_ptr(),
                        jvt.as_ptr(),
                        &m,
                        &n,
                        std::ptr::null_mut(),
                        &m,
                        AsPtr::as_mut_ptr(&mut s),
                        AsPtr::as_mut_ptr(u.as_mut().map(|x| x.as_mut_slice()).unwrap_or(&mut [])),
                        &m,
                        AsPtr::as_mut_ptr(vt.as_mut().map(|x| x.as_mut_slice()).unwrap_or(&mut [])),
                        &n,
                        AsPtr::as_mut_ptr(&mut work_size),
                        &(-1),
                        AsPtr::as_mut_ptr(&mut rwork),
                        &mut info,
                    );
                }
                info.as_lapack_result()?;
                let lwork = work_size[0].to_usize().unwrap();
                let work = vec_uninit(lwork);
                Ok(SvdWork {
                    layout,
                    ju,
                    jvt,
                    s,
                    u,
                    vt,
                    work,
                    rwork: Some(rwork),
                })
            }

            fn calc(&mut self, a: &mut [Self::Elem]) -> Result<SvdRef<Self::Elem>> {
                let m = self.layout.lda();
                let n = self.layout.len();
                let lwork = self.work.len().to_i32().unwrap();

                let mut info = 0;
                unsafe {
                    $svd(
                        self.ju.as_ptr(),
                        self.jvt.as_ptr(),
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
                        &n,
                        AsPtr::as_mut_ptr(&mut self.work),
                        &(lwork as i32),
                        AsPtr::as_mut_ptr(self.rwork.as_mut().unwrap()),
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

                match self.layout {
                    MatrixLayout::F { .. } => Ok(SvdRef { s, u, vt }),
                    MatrixLayout::C { .. } => Ok(SvdRef { s, u: vt, vt: u }),
                }
            }

            fn eval(mut self, a: &mut [Self::Elem]) -> Result<SvdOwned<Self::Elem>> {
                let _ref = self.calc(a)?;
                let s = unsafe { self.s.assume_init() };
                let u = self.u.map(|v| unsafe { v.assume_init() });
                let vt = self.vt.map(|v| unsafe { v.assume_init() });
                match self.layout {
                    MatrixLayout::F { .. } => Ok(SvdOwned { s, u, vt }),
                    MatrixLayout::C { .. } => Ok(SvdOwned { s, u: vt, vt: u }),
                }
            }
        }
    };
}
impl_svd_work_c!(c64, lapack_sys::zgesvd_);
impl_svd_work_c!(c32, lapack_sys::cgesvd_);

macro_rules! impl_svd_work_r {
    ($s:ty, $svd:path) => {
        impl SvdWorkImpl for SvdWork<$s> {
            type Elem = $s;

            fn new(layout: MatrixLayout, calc_u: bool, calc_vt: bool) -> Result<Self> {
                let ju = match layout {
                    MatrixLayout::F { .. } => JobSvd::from_bool(calc_u),
                    MatrixLayout::C { .. } => JobSvd::from_bool(calc_vt),
                };
                let jvt = match layout {
                    MatrixLayout::F { .. } => JobSvd::from_bool(calc_vt),
                    MatrixLayout::C { .. } => JobSvd::from_bool(calc_u),
                };

                let m = layout.lda();
                let mut u = match ju {
                    JobSvd::All => Some(vec_uninit((m * m) as usize)),
                    JobSvd::None => None,
                    _ => unimplemented!("SVD with partial vector output is not supported yet"),
                };

                let n = layout.len();
                let mut vt = match jvt {
                    JobSvd::All => Some(vec_uninit((n * n) as usize)),
                    JobSvd::None => None,
                    _ => unimplemented!("SVD with partial vector output is not supported yet"),
                };

                let k = std::cmp::min(m, n);
                let mut s = vec_uninit(k as usize);

                // eval work size
                let mut info = 0;
                let mut work_size = [Self::Elem::zero()];
                unsafe {
                    $svd(
                        ju.as_ptr(),
                        jvt.as_ptr(),
                        &m,
                        &n,
                        std::ptr::null_mut(),
                        &m,
                        AsPtr::as_mut_ptr(&mut s),
                        AsPtr::as_mut_ptr(u.as_mut().map(|x| x.as_mut_slice()).unwrap_or(&mut [])),
                        &m,
                        AsPtr::as_mut_ptr(vt.as_mut().map(|x| x.as_mut_slice()).unwrap_or(&mut [])),
                        &n,
                        AsPtr::as_mut_ptr(&mut work_size),
                        &(-1),
                        &mut info,
                    );
                }
                info.as_lapack_result()?;
                let lwork = work_size[0].to_usize().unwrap();
                let work = vec_uninit(lwork);
                Ok(SvdWork {
                    layout,
                    ju,
                    jvt,
                    s,
                    u,
                    vt,
                    work,
                    rwork: None,
                })
            }

            fn calc(&mut self, a: &mut [Self::Elem]) -> Result<SvdRef<Self::Elem>> {
                let m = self.layout.lda();
                let n = self.layout.len();
                let lwork = self.work.len().to_i32().unwrap();

                let mut info = 0;
                unsafe {
                    $svd(
                        self.ju.as_ptr(),
                        self.jvt.as_ptr(),
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
                        &n,
                        AsPtr::as_mut_ptr(&mut self.work),
                        &(lwork as i32),
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

                match self.layout {
                    MatrixLayout::F { .. } => Ok(SvdRef { s, u, vt }),
                    MatrixLayout::C { .. } => Ok(SvdRef { s, u: vt, vt: u }),
                }
            }

            fn eval(mut self, a: &mut [Self::Elem]) -> Result<SvdOwned<Self::Elem>> {
                let _ref = self.calc(a)?;
                let s = unsafe { self.s.assume_init() };
                let u = self.u.map(|v| unsafe { v.assume_init() });
                let vt = self.vt.map(|v| unsafe { v.assume_init() });
                match self.layout {
                    MatrixLayout::F { .. } => Ok(SvdOwned { s, u, vt }),
                    MatrixLayout::C { .. } => Ok(SvdOwned { s, u: vt, vt: u }),
                }
            }
        }
    };
}
impl_svd_work_r!(f64, lapack_sys::dgesvd_);
impl_svd_work_r!(f32, lapack_sys::sgesvd_);

macro_rules! impl_svd {
    (@real, $scalar:ty, $gesvd:path) => {
        impl_svd!(@body, $scalar, $gesvd, );
    };
    (@complex, $scalar:ty, $gesvd:path) => {
        impl_svd!(@body, $scalar, $gesvd, rwork);
    };
    (@body, $scalar:ty, $gesvd:path, $($rwork_ident:ident),*) => {
        impl SVD_ for $scalar {
            fn svd(l: MatrixLayout, calc_u: bool, calc_vt: bool, a: &mut [Self],) -> Result<SVDOutput<Self>> {
                let ju = match l {
                    MatrixLayout::F { .. } => JobSvd::from_bool(calc_u),
                    MatrixLayout::C { .. } => JobSvd::from_bool(calc_vt),
                };
                let jvt = match l {
                    MatrixLayout::F { .. } => JobSvd::from_bool(calc_vt),
                    MatrixLayout::C { .. } => JobSvd::from_bool(calc_u),
                };

                let m = l.lda();
                let mut u = match ju {
                    JobSvd::All => Some(vec_uninit( (m * m) as usize)),
                    JobSvd::None => None,
                    _ => unimplemented!("SVD with partial vector output is not supported yet")
                };

                let n = l.len();
                let mut vt = match jvt {
                    JobSvd::All => Some(vec_uninit( (n * n) as usize)),
                    JobSvd::None => None,
                    _ => unimplemented!("SVD with partial vector output is not supported yet")
                };

                let k = std::cmp::min(m, n);
                let mut s = vec_uninit( k as usize);

                $(
                let mut $rwork_ident: Vec<MaybeUninit<Self::Real>> = vec_uninit(5 * k as usize);
                )*

                // eval work size
                let mut info = 0;
                let mut work_size = [Self::zero()];
                unsafe {
                    $gesvd(
                        ju.as_ptr(),
                        jvt.as_ptr(),
                        &m,
                        &n,
                        AsPtr::as_mut_ptr(a),
                        &m,
                        AsPtr::as_mut_ptr(&mut s),
                        AsPtr::as_mut_ptr(u.as_mut().map(|x| x.as_mut_slice()).unwrap_or(&mut [])),
                        &m,
                        AsPtr::as_mut_ptr(vt.as_mut().map(|x| x.as_mut_slice()).unwrap_or(&mut [])),
                        &n,
                        AsPtr::as_mut_ptr(&mut work_size),
                        &(-1),
                        $(AsPtr::as_mut_ptr(&mut $rwork_ident),)*
                        &mut info,
                    );
                }
                info.as_lapack_result()?;

                // calc
                let lwork = work_size[0].to_usize().unwrap();
                let mut work: Vec<MaybeUninit<Self>> = vec_uninit(lwork);
                unsafe {
                    $gesvd(
                        ju.as_ptr(),
                        jvt.as_ptr() ,
                        &m,
                        &n,
                        AsPtr::as_mut_ptr(a),
                        &m,
                        AsPtr::as_mut_ptr(&mut s),
                        AsPtr::as_mut_ptr(u.as_mut().map(|x| x.as_mut_slice()).unwrap_or(&mut [])),
                        &m,
                        AsPtr::as_mut_ptr(vt.as_mut().map(|x| x.as_mut_slice()).unwrap_or(&mut [])),
                        &n,
                        AsPtr::as_mut_ptr(&mut work),
                        &(lwork as i32),
                        $(AsPtr::as_mut_ptr(&mut $rwork_ident),)*
                        &mut info,
                    );
                }
                info.as_lapack_result()?;

                let s = unsafe { s.assume_init() };
                let u = u.map(|v| unsafe { v.assume_init() });
                let vt = vt.map(|v| unsafe { v.assume_init() });

                match l {
                    MatrixLayout::F { .. } => Ok(SVDOutput { s, u, vt }),
                    MatrixLayout::C { .. } => Ok(SVDOutput { s, u: vt, vt: u }),
                }
            }
        }
    };
} // impl_svd!

impl_svd!(@real, f64, lapack_sys::dgesvd_);
impl_svd!(@real, f32, lapack_sys::sgesvd_);
impl_svd!(@complex, c64, lapack_sys::zgesvd_);
impl_svd!(@complex, c32, lapack_sys::cgesvd_);
