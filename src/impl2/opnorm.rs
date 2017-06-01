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

pub trait OperatorNorm_: Sized {
    type Output;

    fn opnorm(NormType, Layout, &[Self]) -> Self::Output;

    fn opnorm_one(l: Layout, a: &[Self]) -> Self::Output {
        Self::opnorm(NormType::One, l, a)
    }

    fn opnorm_inf(l: Layout, a: &[Self]) -> Self::Output {
        Self::opnorm(NormType::Infinity, l, a)
    }

    fn opnorm_fro(l: Layout, a: &[Self]) -> Self::Output {
        Self::opnorm(NormType::Frobenius, l, a)
    }
}

macro_rules! impl_opnorm {
    ($scalar:ty, $output:ty, $lange:path) => {
impl OperatorNorm_ for $scalar {
    type Output = $output;
    fn opnorm(t: NormType, l: Layout, a: &[Self]) -> Self::Output {
        let (m, n) = l.ffi_size();
        $lange(l.ffi_layout(), t as u8, m, n, a, m)
    }
}
}} // impl_opnorm!

impl_opnorm!(f64, f64, c::dlange);
impl_opnorm!(f32, f32, c::slange);
impl_opnorm!(c64, f64, c::zlange);
impl_opnorm!(c32, f32, c::clange);
