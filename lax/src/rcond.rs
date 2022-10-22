//! Reciprocal conditional number

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
