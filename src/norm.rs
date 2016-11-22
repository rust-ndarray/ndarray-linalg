//! Implement Norms for matrices

use lapack::c::*;

pub trait ImplNorm: Sized {
    fn norm_1(m: usize, n: usize, mut a: Vec<Self>) -> Self;
    fn norm_i(m: usize, n: usize, mut a: Vec<Self>) -> Self;
    fn norm_f(m: usize, n: usize, mut a: Vec<Self>) -> Self;
}

macro_rules! impl_norm {
    ($scalar:ty, $lange:path) => {
impl ImplNorm for $scalar {
    fn norm_1(m: usize, n: usize, mut a: Vec<Self>) -> Self {
        $lange(Layout::ColumnMajor, b'o', m as i32, n as i32, &mut a, m as i32)
    }
    fn norm_i(m: usize, n: usize, mut a: Vec<Self>) -> Self {
        $lange(Layout::ColumnMajor, b'i', m as i32, n as i32, &mut a, m as i32)
    }
    fn norm_f(m: usize, n: usize, mut a: Vec<Self>) -> Self {
        $lange(Layout::ColumnMajor, b'f', m as i32, n as i32, &mut a, m as i32)
    }
}
}} // end macro_rules

impl_norm!(f64, dlange);
impl_norm!(f32, slange);
