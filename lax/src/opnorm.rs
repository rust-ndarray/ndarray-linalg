//! Operator norms of matrices

use super::NormType;
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
                let mut work = if matches!(t, NormType::Infinity) {
                    unsafe { vec_uninit(m as usize) }
                } else {
                    Vec::new()
                };
                unsafe { $lange(t as u8, m, n, a, m, &mut work) }
            }
        }
    };
} // impl_opnorm!

impl_opnorm!(f64, lapack::dlange);
impl_opnorm!(f32, lapack::slange);
impl_opnorm!(c64, lapack::zlange);
impl_opnorm!(c32, lapack::clange);
