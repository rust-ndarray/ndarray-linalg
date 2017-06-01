//! Implement Operator norms for matrices

use lapack::c;

use types::*;
use layout::*;

#[repr(u8)]
pub enum NormType {
    One = b'o',
    Infinity = b'i',
    Frobenius = b'f',
}

pub trait OperatorNorm_: AssociatedReal {
    fn opnorm(NormType, Layout, &[Self]) -> Self::Real;
}

macro_rules! impl_opnorm {
    ($scalar:ty, $lange:path) => {
impl OperatorNorm_ for $scalar {
    fn opnorm(t: NormType, l: Layout, a: &[Self]) -> Self::Real {
        let (m, n) = l.ffi_size();
        $lange(l.ffi_layout(), t as u8, m, n, a, m)
    }
}
}} // impl_opnorm!

impl_opnorm!(f64, c::dlange);
impl_opnorm!(f32, c::slange);
impl_opnorm!(c64, c::zlange);
impl_opnorm!(c32, c::clange);
