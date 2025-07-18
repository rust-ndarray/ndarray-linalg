//! Generalized eigenvalue problem for general matrices
//!
//! LAPACK correspondance
//! ----------------------
//!
//! | f32   | f64   | c32   | c64   |
//! |:------|:------|:------|:------|
//! | sggev | dggev | cggev | zggev |
//!
use std::mem::MaybeUninit;

use crate::eig::reconstruct_eigenvectors;
use crate::{error::*, layout::MatrixLayout, *};
use cauchy::*;
use num_traits::{ToPrimitive, Zero};

pub enum GeneralizedEigenvalue<T: Scalar> {
    /// Finite generalized eigenvalue: `Finite(α/β, (α, β))`
    Finite(T, (T, T)),

    /// Indeterminate generalized eigenvalue: `Indeterminate((α, β))`
    Indeterminate((T, T)),
}

#[non_exhaustive]
pub struct EigGeneralizedWork<T: Scalar> {
    /// Problem size
    pub n: i32,
    /// Compute right eigenvectors or not
    pub jobvr: JobEv,
    /// Compute left eigenvectors or not
    pub jobvl: JobEv,

    /// Eigenvalues: alpha (numerators)
    pub alpha: Vec<MaybeUninit<T::Complex>>,
    /// Eigenvalues: beta (denominators)
    pub beta: Vec<MaybeUninit<T::Complex>>,
    /// Real part of alpha (eigenvalue numerators) used in real routines
    pub alpha_re: Option<Vec<MaybeUninit<T::Real>>>,
    /// Imaginary part of alpha (eigenvalue numerators) used in real routines
    pub alpha_im: Option<Vec<MaybeUninit<T::Real>>>,
    /// Real part of beta (eigenvalue denominators) used in real routines
    pub beta_re: Option<Vec<MaybeUninit<T::Real>>>,
    /// Imaginary part of beta (eigenvalue denominators) used in real routines
    pub beta_im: Option<Vec<MaybeUninit<T::Real>>>,

    /// Left eigenvectors
    pub vc_l: Option<Vec<MaybeUninit<T::Complex>>>,
    /// Left eigenvectors used in real routines
    pub vr_l: Option<Vec<MaybeUninit<T::Real>>>,
    /// Right eigenvectors
    pub vc_r: Option<Vec<MaybeUninit<T::Complex>>>,
    /// Right eigenvectors used in real routines
    pub vr_r: Option<Vec<MaybeUninit<T::Real>>>,

    /// Working memory
    pub work: Vec<MaybeUninit<T>>,
    /// Working memory with `T::Real`
    pub rwork: Option<Vec<MaybeUninit<T::Real>>>,
}

impl<T> EigGeneralizedWork<T>
where
    T: Scalar,
    EigGeneralizedWork<T>: EigGeneralizedWorkImpl<Elem = T>,
{
    /// Create new working memory for eigenvalues compution.
    pub fn new(calc_v: bool, l: MatrixLayout) -> Result<Self> {
        EigGeneralizedWorkImpl::new(calc_v, l)
    }

    /// Compute eigenvalues and vectors on this working memory.
    pub fn calc(&mut self, a: &mut [T], b: &mut [T]) -> Result<EigGeneralizedRef<T>> {
        EigGeneralizedWorkImpl::calc(self, a, b)
    }

    /// Compute eigenvalues and vectors by consuming this working memory.
    pub fn eval(self, a: &mut [T], b: &mut [T]) -> Result<EigGeneralizedOwned<T>> {
        EigGeneralizedWorkImpl::eval(self, a, b)
    }
}

/// Owned result of eigenvalue problem by [EigGeneralizedWork::eval]
#[derive(Debug, Clone, PartialEq)]
pub struct EigGeneralizedOwned<T: Scalar> {
    /// Eigenvalues
    pub alpha: Vec<T::Complex>,

    pub beta: Vec<T::Complex>,

    /// Right eigenvectors
    pub vr: Option<Vec<T::Complex>>,

    /// Left eigenvectors
    pub vl: Option<Vec<T::Complex>>,
}

/// Reference result of eigenvalue problem by [EigGeneralizedWork::calc]
#[derive(Debug, Clone, PartialEq)]
pub struct EigGeneralizedRef<'work, T: Scalar> {
    /// Eigenvalues
    pub alpha: &'work [T::Complex],

    pub beta: &'work [T::Complex],

    /// Right eigenvectors
    pub vr: Option<&'work [T::Complex]>,

    /// Left eigenvectors
    pub vl: Option<&'work [T::Complex]>,
}

/// Helper trait for implementing [EigGeneralizedWork] methods
pub trait EigGeneralizedWorkImpl: Sized {
    type Elem: Scalar;
    fn new(calc_v: bool, l: MatrixLayout) -> Result<Self>;
    fn calc<'work>(
        &'work mut self,
        a: &mut [Self::Elem],
        b: &mut [Self::Elem],
    ) -> Result<EigGeneralizedRef<'work, Self::Elem>>;
    fn eval(
        self,
        a: &mut [Self::Elem],
        b: &mut [Self::Elem],
    ) -> Result<EigGeneralizedOwned<Self::Elem>>;
}

macro_rules! impl_eig_generalized_work_c {
    ($f:ty, $c:ty, $ev:path) => {
        impl EigGeneralizedWorkImpl for EigGeneralizedWork<$c> {
            type Elem = $c;

            fn new(calc_v: bool, l: MatrixLayout) -> Result<Self> {
                let (n, _) = l.size();
                let (jobvl, jobvr) = if calc_v {
                    match l {
                        MatrixLayout::C { .. } => (JobEv::All, JobEv::None),
                        MatrixLayout::F { .. } => (JobEv::None, JobEv::All),
                    }
                } else {
                    (JobEv::None, JobEv::None)
                };
                let mut rwork = vec_uninit(8 * n as usize);

                let mut alpha = vec_uninit(n as usize);
                let mut beta = vec_uninit(n as usize);

                let mut vc_l = jobvl.then(|| vec_uninit((n * n) as usize));
                let mut vc_r = jobvr.then(|| vec_uninit((n * n) as usize));

                // calc work size
                let mut info = 0;
                let mut work_size = [<$c>::zero()];
                unsafe {
                    $ev(
                        jobvl.as_ptr(),
                        jobvr.as_ptr(),
                        &n,
                        std::ptr::null_mut(),
                        &n,
                        std::ptr::null_mut(),
                        &n,
                        AsPtr::as_mut_ptr(&mut alpha),
                        AsPtr::as_mut_ptr(&mut beta),
                        AsPtr::as_mut_ptr(vc_l.as_deref_mut().unwrap_or(&mut [])),
                        &n,
                        AsPtr::as_mut_ptr(vc_r.as_deref_mut().unwrap_or(&mut [])),
                        &n,
                        AsPtr::as_mut_ptr(&mut work_size),
                        &(-1),
                        AsPtr::as_mut_ptr(&mut rwork),
                        &mut info,
                    )
                };
                info.as_lapack_result()?;

                let lwork = work_size[0].to_usize().unwrap();
                let work: Vec<MaybeUninit<$c>> = vec_uninit(lwork);
                Ok(Self {
                    n,
                    jobvl,
                    jobvr,
                    alpha,
                    beta,
                    alpha_re: None,
                    alpha_im: None,
                    beta_re: None,
                    beta_im: None,
                    rwork: Some(rwork),
                    vc_l,
                    vc_r,
                    vr_l: None,
                    vr_r: None,
                    work,
                })
            }

            fn calc<'work>(
                &'work mut self,
                a: &mut [Self::Elem],
                b: &mut [Self::Elem],
            ) -> Result<EigGeneralizedRef<'work, Self::Elem>> {
                let lwork = self.work.len().to_i32().unwrap();
                let mut info = 0;
                unsafe {
                    $ev(
                        self.jobvl.as_ptr(),
                        self.jobvr.as_ptr(),
                        &self.n,
                        AsPtr::as_mut_ptr(a),
                        &self.n,
                        AsPtr::as_mut_ptr(b),
                        &self.n,
                        AsPtr::as_mut_ptr(&mut self.alpha),
                        AsPtr::as_mut_ptr(&mut self.beta),
                        AsPtr::as_mut_ptr(self.vc_l.as_deref_mut().unwrap_or(&mut [])),
                        &self.n,
                        AsPtr::as_mut_ptr(self.vc_r.as_deref_mut().unwrap_or(&mut [])),
                        &self.n,
                        AsPtr::as_mut_ptr(&mut self.work),
                        &lwork,
                        AsPtr::as_mut_ptr(self.rwork.as_mut().unwrap()),
                        &mut info,
                    )
                };
                info.as_lapack_result()?;
                // Hermite conjugate
                if let Some(vl) = self.vc_l.as_mut() {
                    for value in vl {
                        let value = unsafe { value.assume_init_mut() };
                        value.im = -value.im;
                    }
                }
                Ok(EigGeneralizedRef {
                    alpha: unsafe { self.alpha.slice_assume_init_ref() },
                    beta: unsafe { self.beta.slice_assume_init_ref() },
                    vl: self
                        .vc_l
                        .as_ref()
                        .map(|v| unsafe { v.slice_assume_init_ref() }),
                    vr: self
                        .vc_r
                        .as_ref()
                        .map(|v| unsafe { v.slice_assume_init_ref() }),
                })
            }

            fn eval(
                mut self,
                a: &mut [Self::Elem],
                b: &mut [Self::Elem],
            ) -> Result<EigGeneralizedOwned<Self::Elem>> {
                let _eig_generalized_ref = self.calc(a, b)?;
                Ok(EigGeneralizedOwned {
                    alpha: unsafe { self.alpha.assume_init() },
                    beta: unsafe { self.beta.assume_init() },
                    vl: self.vc_l.map(|v| unsafe { v.assume_init() }),
                    vr: self.vc_r.map(|v| unsafe { v.assume_init() }),
                })
            }
        }

        impl EigGeneralizedOwned<$c> {
            pub fn calc_eigs(&self, thresh_opt: Option<$f>) -> Vec<GeneralizedEigenvalue<$c>> {
                self.alpha
                    .iter()
                    .zip(self.beta.iter())
                    .map(|(alpha, beta)| {
                        if let Some(thresh) = thresh_opt {
                            if beta.abs() < thresh {
                                GeneralizedEigenvalue::Indeterminate((alpha.clone(), beta.clone()))
                            } else {
                                GeneralizedEigenvalue::Finite(
                                    alpha / beta,
                                    (alpha.clone(), beta.clone()),
                                )
                            }
                        } else {
                            if beta.is_zero() {
                                GeneralizedEigenvalue::Indeterminate((alpha.clone(), beta.clone()))
                            } else {
                                GeneralizedEigenvalue::Finite(
                                    alpha / beta,
                                    (alpha.clone(), beta.clone()),
                                )
                            }
                        }
                    })
                    .collect::<Vec<_>>()
            }
        }
    };
}

impl_eig_generalized_work_c!(f32, c32, lapack_sys::cggev_);
impl_eig_generalized_work_c!(f64, c64, lapack_sys::zggev_);

macro_rules! impl_eig_generalized_work_r {
    ($f:ty, $c:ty, $ev:path) => {
        impl EigGeneralizedWorkImpl for EigGeneralizedWork<$f> {
            type Elem = $f;

            fn new(calc_v: bool, l: MatrixLayout) -> Result<Self> {
                let (n, _) = l.size();
                let (jobvl, jobvr) = if calc_v {
                    match l {
                        MatrixLayout::C { .. } => (JobEv::All, JobEv::None),
                        MatrixLayout::F { .. } => (JobEv::None, JobEv::All),
                    }
                } else {
                    (JobEv::None, JobEv::None)
                };
                let mut alpha_re = vec_uninit(n as usize);
                let mut alpha_im = vec_uninit(n as usize);
                let mut beta_re = vec_uninit(n as usize);
                let mut vr_l = jobvl.then(|| vec_uninit((n * n) as usize));
                let mut vr_r = jobvr.then(|| vec_uninit((n * n) as usize));
                let vc_l = jobvl.then(|| vec_uninit((n * n) as usize));
                let vc_r = jobvr.then(|| vec_uninit((n * n) as usize));

                // calc work size
                let mut info = 0;
                let mut work_size: [$f; 1] = [0.0];
                unsafe {
                    $ev(
                        jobvl.as_ptr(),
                        jobvr.as_ptr(),
                        &n,
                        std::ptr::null_mut(),
                        &n,
                        std::ptr::null_mut(),
                        &n,
                        AsPtr::as_mut_ptr(&mut alpha_re),
                        AsPtr::as_mut_ptr(&mut alpha_im),
                        AsPtr::as_mut_ptr(&mut beta_re),
                        AsPtr::as_mut_ptr(vr_l.as_deref_mut().unwrap_or(&mut [])),
                        &n,
                        AsPtr::as_mut_ptr(vr_r.as_deref_mut().unwrap_or(&mut [])),
                        &n,
                        AsPtr::as_mut_ptr(&mut work_size),
                        &(-1),
                        &mut info,
                    )
                };
                info.as_lapack_result()?;

                // actual ev
                let lwork = work_size[0].to_usize().unwrap();
                let work = vec_uninit(lwork);

                Ok(Self {
                    n,
                    jobvr,
                    jobvl,
                    alpha: vec_uninit(n as usize),
                    beta: vec_uninit(n as usize),
                    alpha_re: Some(alpha_re),
                    alpha_im: Some(alpha_im),
                    beta_re: Some(beta_re),
                    beta_im: None,
                    rwork: None,
                    vr_l,
                    vr_r,
                    vc_l,
                    vc_r,
                    work,
                })
            }

            fn calc<'work>(
                &'work mut self,
                a: &mut [Self::Elem],
                b: &mut [Self::Elem],
            ) -> Result<EigGeneralizedRef<'work, Self::Elem>> {
                let lwork = self.work.len().to_i32().unwrap();
                let mut info = 0;
                unsafe {
                    $ev(
                        self.jobvl.as_ptr(),
                        self.jobvr.as_ptr(),
                        &self.n,
                        AsPtr::as_mut_ptr(a),
                        &self.n,
                        AsPtr::as_mut_ptr(b),
                        &self.n,
                        AsPtr::as_mut_ptr(self.alpha_re.as_mut().unwrap()),
                        AsPtr::as_mut_ptr(self.alpha_im.as_mut().unwrap()),
                        AsPtr::as_mut_ptr(self.beta_re.as_mut().unwrap()),
                        AsPtr::as_mut_ptr(self.vr_l.as_deref_mut().unwrap_or(&mut [])),
                        &self.n,
                        AsPtr::as_mut_ptr(self.vr_r.as_deref_mut().unwrap_or(&mut [])),
                        &self.n,
                        AsPtr::as_mut_ptr(&mut self.work),
                        &lwork,
                        &mut info,
                    )
                };
                info.as_lapack_result()?;

                let alpha_re = self
                    .alpha_re
                    .as_ref()
                    .map(|e| unsafe { e.slice_assume_init_ref() })
                    .unwrap();
                let alpha_im = self
                    .alpha_im
                    .as_ref()
                    .map(|e| unsafe { e.slice_assume_init_ref() })
                    .unwrap();
                let beta_re = self
                    .beta_re
                    .as_ref()
                    .map(|e| unsafe { e.slice_assume_init_ref() })
                    .unwrap();
                reconstruct_eigs_optional_im(alpha_re, Some(alpha_im), &mut self.alpha);
                reconstruct_eigs_optional_im(beta_re, None, &mut self.beta);

                if let Some(v) = self.vr_l.as_ref() {
                    let v = unsafe { v.slice_assume_init_ref() };
                    reconstruct_eigenvectors(true, alpha_im, v, self.vc_l.as_mut().unwrap());
                }
                if let Some(v) = self.vr_r.as_ref() {
                    let v = unsafe { v.slice_assume_init_ref() };
                    reconstruct_eigenvectors(false, alpha_im, v, self.vc_r.as_mut().unwrap());
                }

                Ok(EigGeneralizedRef {
                    alpha: unsafe { self.alpha.slice_assume_init_ref() },
                    beta: unsafe { self.beta.slice_assume_init_ref() },
                    vl: self
                        .vc_l
                        .as_ref()
                        .map(|v| unsafe { v.slice_assume_init_ref() }),
                    vr: self
                        .vc_r
                        .as_ref()
                        .map(|v| unsafe { v.slice_assume_init_ref() }),
                })
            }

            fn eval(
                mut self,
                a: &mut [Self::Elem],
                b: &mut [Self::Elem],
            ) -> Result<EigGeneralizedOwned<Self::Elem>> {
                let _eig_generalized_ref = self.calc(a, b)?;
                Ok(EigGeneralizedOwned {
                    alpha: unsafe { self.alpha.assume_init() },
                    beta: unsafe { self.beta.assume_init() },
                    vl: self.vc_l.map(|v| unsafe { v.assume_init() }),
                    vr: self.vc_r.map(|v| unsafe { v.assume_init() }),
                })
            }
        }

        impl EigGeneralizedOwned<$f> {
            pub fn calc_eigs(&self, thresh_opt: Option<$f>) -> Vec<GeneralizedEigenvalue<$c>> {
                self.alpha
                    .iter()
                    .zip(self.beta.iter())
                    .map(|(alpha, beta)| {
                        if let Some(thresh) = thresh_opt {
                            if beta.abs() < thresh {
                                GeneralizedEigenvalue::Indeterminate((alpha.clone(), beta.clone()))
                            } else {
                                GeneralizedEigenvalue::Finite(alpha / beta, (alpha.clone(), beta.clone()))
                            }
                        } else {
                            if beta.is_zero() {
                                GeneralizedEigenvalue::Indeterminate((alpha.clone(), beta.clone()))
                            } else {
                                GeneralizedEigenvalue::Finite(alpha / beta, (alpha.clone(), beta.clone()))
                            }
                        }
                    })
                    .collect::<Vec<_>>()
            }
        }
    };
}
impl_eig_generalized_work_r!(f32, c32, lapack_sys::sggev_);
impl_eig_generalized_work_r!(f64, c64, lapack_sys::dggev_);

/// Create complex eigenvalues from real and optional imaginary parts.
fn reconstruct_eigs_optional_im<T: Scalar>(
    re: &[T],
    im_opt: Option<&[T]>,
    eigs: &mut [MaybeUninit<T::Complex>],
) {
    let n = eigs.len();
    assert_eq!(re.len(), n);

    if let Some(im) = im_opt {
        assert_eq!(im.len(), n);
        for i in 0..n {
            eigs[i].write(T::complex(re[i], im[i]));
        }
    } else {
        for i in 0..n {
            eigs[i].write(T::complex(re[i], T::zero()));
        }
    }
}
