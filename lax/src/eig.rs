//! Eigenvalue problem for general matricies

use crate::{error::*, layout::MatrixLayout, *};
use cauchy::*;
use num_traits::{ToPrimitive, Zero};

#[cfg_attr(doc, katexit::katexit)]
/// Eigenvalue problem for general matrix
///
/// To manage memory more strictly, use [EigWork].
///
/// Right and Left eigenvalue problem
/// ----------------------------------
/// LAPACK can solve both right eigenvalue problem
/// $$
/// AV_R = V_R \Lambda
/// $$
/// where $V_R = \left( v_R^1, \cdots, v_R^n \right)$ are right eigenvectors
/// and left eigenvalue problem
/// $$
/// V_L^\dagger A = V_L^\dagger \Lambda
/// $$
/// where $V_L = \left( v_L^1, \cdots, v_L^n \right)$ are left eigenvectors
/// and eigenvalues
/// $$
/// \Lambda = \begin{pmatrix}
///   \lambda_1 &        & 0 \\\\
///             & \ddots &   \\\\
///   0         &        & \lambda_n
/// \end{pmatrix}
/// $$
/// which satisfies $A v_R^i = \lambda_i v_R^i$ and
/// $\left(v_L^i\right)^\dagger A = \lambda_i \left(v_L^i\right)^\dagger$
/// for column-major matrices, although row-major matrices are not supported.
/// Since a row-major matrix can be interpreted
/// as a transpose of a column-major matrix,
/// this transforms right eigenvalue problem to left one:
///
/// $$
/// A^\dagger V = V Λ ⟺ V^\dagger A = Λ V^\dagger
/// $$
///
#[non_exhaustive]
pub struct EigWork<T: Scalar> {
    /// Problem size
    pub n: i32,
    /// Compute right eigenvectors or not
    pub jobvr: JobEv,
    /// Compute left eigenvectors or not
    pub jobvl: JobEv,

    /// Eigenvalues
    pub eigs: Vec<MaybeUninit<T::Complex>>,
    /// Real part of eigenvalues used in real routines
    pub eigs_re: Option<Vec<MaybeUninit<T::Real>>>,
    /// Imaginary part of eigenvalues used in real routines
    pub eigs_im: Option<Vec<MaybeUninit<T::Real>>>,

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

impl<T> EigWork<T>
where
    T: Scalar,
    EigWork<T>: EigWorkImpl<Elem = T>,
{
    /// Create new working memory for eigenvalues compution.
    pub fn new(calc_v: bool, l: MatrixLayout) -> Result<Self> {
        EigWorkImpl::new(calc_v, l)
    }

    /// Compute eigenvalues and vectors on this working memory.
    pub fn calc(&mut self, a: &mut [T]) -> Result<EigRef<T>> {
        EigWorkImpl::calc(self, a)
    }

    /// Compute eigenvalues and vectors by consuming this working memory.
    pub fn eval(self, a: &mut [T]) -> Result<EigOwned<T>> {
        EigWorkImpl::eval(self, a)
    }
}

/// Owned result of eigenvalue problem by [EigWork::eval]
#[derive(Debug, Clone, PartialEq)]
pub struct EigOwned<T: Scalar> {
    /// Eigenvalues
    pub eigs: Vec<T::Complex>,
    /// Right eigenvectors
    pub vr: Option<Vec<T::Complex>>,
    /// Left eigenvectors
    pub vl: Option<Vec<T::Complex>>,
}

/// Reference result of eigenvalue problem by [EigWork::calc]
#[derive(Debug, Clone, PartialEq)]
pub struct EigRef<'work, T: Scalar> {
    /// Eigenvalues
    pub eigs: &'work [T::Complex],
    /// Right eigenvectors
    pub vr: Option<&'work [T::Complex]>,
    /// Left eigenvectors
    pub vl: Option<&'work [T::Complex]>,
}

/// Helper trait for implementing [EigWork] methods
pub trait EigWorkImpl: Sized {
    type Elem: Scalar;
    fn new(calc_v: bool, l: MatrixLayout) -> Result<Self>;
    fn calc<'work>(&'work mut self, a: &mut [Self::Elem]) -> Result<EigRef<'work, Self::Elem>>;
    fn eval(self, a: &mut [Self::Elem]) -> Result<EigOwned<Self::Elem>>;
}

macro_rules! impl_eig_work_c {
    ($c:ty, $ev:path) => {
        impl EigWorkImpl for EigWork<$c> {
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
                let mut eigs = vec_uninit(n as usize);
                let mut rwork = vec_uninit(2 * n as usize);

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
                        AsPtr::as_mut_ptr(&mut eigs),
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
                    eigs,
                    eigs_re: None,
                    eigs_im: None,
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
            ) -> Result<EigRef<'work, Self::Elem>> {
                let lwork = self.work.len().to_i32().unwrap();
                let mut info = 0;
                unsafe {
                    $ev(
                        self.jobvl.as_ptr(),
                        self.jobvr.as_ptr(),
                        &self.n,
                        AsPtr::as_mut_ptr(a),
                        &self.n,
                        AsPtr::as_mut_ptr(&mut self.eigs),
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
                Ok(EigRef {
                    eigs: unsafe { self.eigs.slice_assume_init_ref() },
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

            fn eval(mut self, a: &mut [Self::Elem]) -> Result<EigOwned<Self::Elem>> {
                let _eig_ref = self.calc(a)?;
                Ok(EigOwned {
                    eigs: unsafe { self.eigs.assume_init() },
                    vl: self.vc_l.map(|v| unsafe { v.assume_init() }),
                    vr: self.vc_r.map(|v| unsafe { v.assume_init() }),
                })
            }
        }
    };
}

impl_eig_work_c!(c32, lapack_sys::cgeev_);
impl_eig_work_c!(c64, lapack_sys::zgeev_);

macro_rules! impl_eig_work_r {
    ($f:ty, $ev:path) => {
        impl EigWorkImpl for EigWork<$f> {
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
                let mut eigs_re = vec_uninit(n as usize);
                let mut eigs_im = vec_uninit(n as usize);
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
                        AsPtr::as_mut_ptr(&mut eigs_re),
                        AsPtr::as_mut_ptr(&mut eigs_im),
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
                    eigs: vec_uninit(n as usize),
                    eigs_re: Some(eigs_re),
                    eigs_im: Some(eigs_im),
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
            ) -> Result<EigRef<'work, Self::Elem>> {
                let lwork = self.work.len().to_i32().unwrap();
                let mut info = 0;
                unsafe {
                    $ev(
                        self.jobvl.as_ptr(),
                        self.jobvr.as_ptr(),
                        &self.n,
                        AsPtr::as_mut_ptr(a),
                        &self.n,
                        AsPtr::as_mut_ptr(self.eigs_re.as_mut().unwrap()),
                        AsPtr::as_mut_ptr(self.eigs_im.as_mut().unwrap()),
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

                let eigs_re = self
                    .eigs_re
                    .as_ref()
                    .map(|e| unsafe { e.slice_assume_init_ref() })
                    .unwrap();
                let eigs_im = self
                    .eigs_im
                    .as_ref()
                    .map(|e| unsafe { e.slice_assume_init_ref() })
                    .unwrap();
                reconstruct_eigs(eigs_re, eigs_im, &mut self.eigs);

                if let Some(v) = self.vr_l.as_ref() {
                    let v = unsafe { v.slice_assume_init_ref() };
                    reconstruct_eigenvectors(true, eigs_im, v, self.vc_l.as_mut().unwrap());
                }
                if let Some(v) = self.vr_r.as_ref() {
                    let v = unsafe { v.slice_assume_init_ref() };
                    reconstruct_eigenvectors(false, eigs_im, v, self.vc_r.as_mut().unwrap());
                }

                Ok(EigRef {
                    eigs: unsafe { self.eigs.slice_assume_init_ref() },
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

            fn eval(mut self, a: &mut [Self::Elem]) -> Result<EigOwned<Self::Elem>> {
                let _eig_ref = self.calc(a)?;
                Ok(EigOwned {
                    eigs: unsafe { self.eigs.assume_init() },
                    vl: self.vc_l.map(|v| unsafe { v.assume_init() }),
                    vr: self.vc_r.map(|v| unsafe { v.assume_init() }),
                })
            }
        }
    };
}
impl_eig_work_r!(f32, lapack_sys::sgeev_);
impl_eig_work_r!(f64, lapack_sys::dgeev_);

/// Reconstruct eigenvectors into complex-array
///
/// From LAPACK API https://software.intel.com/en-us/node/469230
///
/// - If the j-th eigenvalue is real,
///   - v(j) = VR(:,j), the j-th column of VR.
///
/// - If the j-th and (j+1)-st eigenvalues form a complex conjugate pair,
///   - v(j)   = VR(:,j) + i*VR(:,j+1)
///   - v(j+1) = VR(:,j) - i*VR(:,j+1).
///
/// In the C-layout case, we need the conjugates of the left
/// eigenvectors, so the signs should be reversed.
fn reconstruct_eigenvectors<T: Scalar>(
    take_hermite_conjugate: bool,
    eig_im: &[T],
    vr: &[T],
    vc: &mut [MaybeUninit<T::Complex>],
) {
    let n = eig_im.len();
    assert_eq!(vr.len(), n * n);
    assert_eq!(vc.len(), n * n);

    let mut col = 0;
    while col < n {
        if eig_im[col].is_zero() {
            // The corresponding eigenvalue is real.
            for row in 0..n {
                let re = vr[row + col * n];
                vc[row + col * n].write(T::complex(re, T::zero()));
            }
            col += 1;
        } else {
            // This is a complex conjugate pair.
            assert!(col + 1 < n);
            for row in 0..n {
                let re = vr[row + col * n];
                let mut im = vr[row + (col + 1) * n];
                if take_hermite_conjugate {
                    im = -im;
                }
                vc[row + col * n].write(T::complex(re, im));
                vc[row + (col + 1) * n].write(T::complex(re, -im));
            }
            col += 2;
        }
    }
}

/// Create complex eigenvalues from real and imaginary parts.
fn reconstruct_eigs<T: Scalar>(re: &[T], im: &[T], eigs: &mut [MaybeUninit<T::Complex>]) {
    let n = eigs.len();
    assert_eq!(re.len(), n);
    assert_eq!(im.len(), n);
    for i in 0..n {
        eigs[i].write(T::complex(re[i], im[i]));
    }
}
