//! Operator norms of matrices

use lapack::c;
use lapack::c::Layout::ColumnMajor as cm;

use layout::MatrixLayout;
use types::*;

pub use super::NormType;

pub trait OperatorNorm_: AssociatedReal {
    unsafe fn opnorm(NormType, MatrixLayout, &[Self]) -> Self::Real;
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
}} // impl_opnorm!

impl_opnorm!(f64, c::dlange);
impl_opnorm!(f32, c::slange);
impl_opnorm!(c64, c::zlange);
impl_opnorm!(c32, c::clange);
