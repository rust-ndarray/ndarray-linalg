//! Operator norms of matrices

use lapacke;
use lapacke::Layout::ColumnMajor as cm;

use crate::layout::MatrixLayout;
use crate::types::*;

pub use super::NormType;

pub trait OperatorNorm_: Scalar {
    unsafe fn opnorm(t: NormType, l: MatrixLayout, a: &[Self]) -> Self::Real;
}

macro_rules! impl_opnorm {
    ($scalar:ty, $lange:path) => {
        impl OperatorNorm_ for $scalar {
            unsafe fn opnorm(t: NormType, l: MatrixLayout, a: &[Self]) -> Self::Real {
                match l {
                    MatrixLayout::F((col, lda)) => $lange(cm, t as u8, lda, col, a, lda),
                    MatrixLayout::C((row, lda)) => $lange(cm, t.transpose() as u8, lda, row, a, lda),
                }
            }
        }
    };
} // impl_opnorm!

impl_opnorm!(f64, lapacke::dlange);
impl_opnorm!(f32, lapacke::slange);
impl_opnorm!(c64, lapacke::zlange);
impl_opnorm!(c32, lapacke::clange);
