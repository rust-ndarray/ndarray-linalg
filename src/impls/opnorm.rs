//! Implement Norms for matrices

use lapack::c::*;

pub trait ImplOpNorm: Sized {
    fn opnorm_1(m: usize, n: usize, a: Vec<Self>) -> Self;
    fn opnorm_i(m: usize, n: usize, a: Vec<Self>) -> Self;
    fn opnorm_f(m: usize, n: usize, a: Vec<Self>) -> Self;
}

macro_rules! impl_opnorm {
    ($scalar:ty, $lange:path) => {
impl ImplOpNorm for $scalar {
    fn opnorm_1(m: usize, n: usize, mut a: Vec<Self>) -> Self {
        $lange(Layout::ColumnMajor, b'o', m as i32, n as i32, &mut a, m as i32)
    }
    fn opnorm_i(m: usize, n: usize, mut a: Vec<Self>) -> Self {
        $lange(Layout::ColumnMajor, b'i', m as i32, n as i32, &mut a, m as i32)
    }
    fn opnorm_f(m: usize, n: usize, mut a: Vec<Self>) -> Self {
        $lange(Layout::ColumnMajor, b'f', m as i32, n as i32, &mut a, m as i32)
    }
}
}} // end macro_rules

impl_opnorm!(f64, dlange);
impl_opnorm!(f32, slange);
