//! Operator norms of matrices

use super::{AsPtr, NormType};
use crate::{layout::MatrixLayout, *};
use cauchy::*;

pub trait OperatorNorm_: Scalar {
    fn opnorm(t: NormType, l: MatrixLayout, a: &[Self]) -> Self::Real;
}

macro_rules! impl_opnorm {
    ($scalar:ty, $lange:path) => {
        impl OperatorNorm_ for $scalar {
            fn opnorm(t: NormType, l: MatrixLayout, a: &[Self]) -> Self::Real {
                let m = l.lda();
                let n = l.len();
                let t = match l {
                    MatrixLayout::F { .. } => t,
                    MatrixLayout::C { .. } => t.transpose(),
                };
                let mut work: Vec<MaybeUninit<Self::Real>> = if matches!(t, NormType::Infinity) {
                    unsafe { vec_uninit(m as usize) }
                } else {
                    Vec::new()
                };
                unsafe {
                    $lange(
                        t.as_ptr(),
                        &m,
                        &n,
                        AsPtr::as_ptr(a),
                        &m,
                        AsPtr::as_mut_ptr(&mut work),
                    )
                }
            }
        }
    };
} // impl_opnorm!

impl_opnorm!(f64, lapack_sys::dlange_);
impl_opnorm!(f32, lapack_sys::slange_);
impl_opnorm!(c64, lapack_sys::zlange_);
impl_opnorm!(c32, lapack_sys::clange_);
