use crate::{error::*, layout::MatrixLayout, *};
use cauchy::*;
use num_traits::Zero;

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
