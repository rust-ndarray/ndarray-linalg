use crate::{error::*, layout::MatrixLayout, *};
use cauchy::*;
use num_traits::Zero;

pub struct RcondWork<T: Scalar> {
    pub layout: MatrixLayout,
    pub work: Vec<MaybeUninit<T>>,
    pub rwork: Option<Vec<MaybeUninit<T::Real>>>,
    pub iwork: Option<Vec<MaybeUninit<i32>>>,
}

pub trait RcondWorkImpl {
    type Elem: Scalar;
    fn new(l: MatrixLayout) -> Self;
    fn calc(
        &mut self,
        a: &[Self::Elem],
        anorm: <Self::Elem as Scalar>::Real,
    ) -> Result<<Self::Elem as Scalar>::Real>;
}

macro_rules! impl_rcond_work_c {
    ($c:ty, $con:path) => {
        impl RcondWorkImpl for RcondWork<$c> {
            type Elem = $c;

            fn new(layout: MatrixLayout) -> Self {
                let (n, _) = layout.size();
                let work = vec_uninit(2 * n as usize);
                let rwork = vec_uninit(2 * n as usize);
                RcondWork {
                    layout,
                    work,
                    rwork: Some(rwork),
                    iwork: None,
                }
            }

            fn calc(
                &mut self,
                a: &[Self::Elem],
                anorm: <Self::Elem as Scalar>::Real,
            ) -> Result<<Self::Elem as Scalar>::Real> {
                let (n, _) = self.layout.size();
                let mut rcond = <Self::Elem as Scalar>::Real::zero();
                let mut info = 0;
                let norm_type = match self.layout {
                    MatrixLayout::C { .. } => NormType::Infinity,
                    MatrixLayout::F { .. } => NormType::One,
                };
                unsafe {
                    $con(
                        norm_type.as_ptr(),
                        &n,
                        AsPtr::as_ptr(a),
                        &self.layout.lda(),
                        &anorm,
                        &mut rcond,
                        AsPtr::as_mut_ptr(&mut self.work),
                        AsPtr::as_mut_ptr(self.rwork.as_mut().unwrap()),
                        &mut info,
                    )
                };
                info.as_lapack_result()?;
                Ok(rcond)
            }
        }
    };
}
impl_rcond_work_c!(c64, lapack_sys::zgecon_);
impl_rcond_work_c!(c32, lapack_sys::cgecon_);

macro_rules! impl_rcond_work_r {
    ($r:ty, $con:path) => {
        impl RcondWorkImpl for RcondWork<$r> {
            type Elem = $r;

            fn new(layout: MatrixLayout) -> Self {
                let (n, _) = layout.size();
                let work = vec_uninit(4 * n as usize);
                let iwork = vec_uninit(n as usize);
                RcondWork {
                    layout,
                    work,
                    rwork: None,
                    iwork: Some(iwork),
                }
            }

            fn calc(
                &mut self,
                a: &[Self::Elem],
                anorm: <Self::Elem as Scalar>::Real,
            ) -> Result<<Self::Elem as Scalar>::Real> {
                let (n, _) = self.layout.size();
                let mut rcond = <Self::Elem as Scalar>::Real::zero();
                let mut info = 0;
                let norm_type = match self.layout {
                    MatrixLayout::C { .. } => NormType::Infinity,
                    MatrixLayout::F { .. } => NormType::One,
                };
                unsafe {
                    $con(
                        norm_type.as_ptr(),
                        &n,
                        AsPtr::as_ptr(a),
                        &self.layout.lda(),
                        &anorm,
                        &mut rcond,
                        AsPtr::as_mut_ptr(&mut self.work),
                        AsPtr::as_mut_ptr(self.iwork.as_mut().unwrap()),
                        &mut info,
                    )
                };
                info.as_lapack_result()?;
                Ok(rcond)
            }
        }
    };
}
impl_rcond_work_r!(f64, lapack_sys::dgecon_);
impl_rcond_work_r!(f32, lapack_sys::sgecon_);

pub trait Rcond_: Scalar + Sized {
    /// Estimates the the reciprocal of the condition number of the matrix in 1-norm.
    ///
    /// `anorm` should be the 1-norm of the matrix `a`.
    fn rcond(l: MatrixLayout, a: &[Self], anorm: Self::Real) -> Result<Self::Real>;
}

macro_rules! impl_rcond_real {
    ($scalar:ty, $gecon:path) => {
        impl Rcond_ for $scalar {
            fn rcond(l: MatrixLayout, a: &[Self], anorm: Self::Real) -> Result<Self::Real> {
                let (n, _) = l.size();
                let mut rcond = Self::Real::zero();
                let mut info = 0;

                let mut work: Vec<MaybeUninit<Self>> = vec_uninit(4 * n as usize);
                let mut iwork: Vec<MaybeUninit<i32>> = vec_uninit(n as usize);
                let norm_type = match l {
                    MatrixLayout::C { .. } => NormType::Infinity,
                    MatrixLayout::F { .. } => NormType::One,
                };
                unsafe {
                    $gecon(
                        norm_type.as_ptr(),
                        &n,
                        AsPtr::as_ptr(a),
                        &l.lda(),
                        &anorm,
                        &mut rcond,
                        AsPtr::as_mut_ptr(&mut work),
                        AsPtr::as_mut_ptr(&mut iwork),
                        &mut info,
                    )
                };
                info.as_lapack_result()?;

                Ok(rcond)
            }
        }
    };
}

impl_rcond_real!(f32, lapack_sys::sgecon_);
impl_rcond_real!(f64, lapack_sys::dgecon_);

macro_rules! impl_rcond_complex {
    ($scalar:ty, $gecon:path) => {
        impl Rcond_ for $scalar {
            fn rcond(l: MatrixLayout, a: &[Self], anorm: Self::Real) -> Result<Self::Real> {
                let (n, _) = l.size();
                let mut rcond = Self::Real::zero();
                let mut info = 0;
                let mut work: Vec<MaybeUninit<Self>> = vec_uninit(2 * n as usize);
                let mut rwork: Vec<MaybeUninit<Self::Real>> = vec_uninit(2 * n as usize);
                let norm_type = match l {
                    MatrixLayout::C { .. } => NormType::Infinity,
                    MatrixLayout::F { .. } => NormType::One,
                };
                unsafe {
                    $gecon(
                        norm_type.as_ptr(),
                        &n,
                        AsPtr::as_ptr(a),
                        &l.lda(),
                        &anorm,
                        &mut rcond,
                        AsPtr::as_mut_ptr(&mut work),
                        AsPtr::as_mut_ptr(&mut rwork),
                        &mut info,
                    )
                };
                info.as_lapack_result()?;

                Ok(rcond)
            }
        }
    };
}

impl_rcond_complex!(c32, lapack_sys::cgecon_);
impl_rcond_complex!(c64, lapack_sys::zgecon_);
