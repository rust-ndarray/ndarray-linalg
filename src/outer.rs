
use blas::c::{Layout, dger, sger};

pub trait ImplOuter: Sized {
    fn outer(m: usize, n: usize, a: &[Self], b: &[Self], ab: &mut [Self]);
}

macro_rules! impl_cholesky {
    ($scalar:ty, $ger:path) => {
impl ImplOuter for $scalar {
    fn outer(m: usize, n: usize, a: &[Self], b: &[Self], mut ab: &mut [Self]) {
        $ger(Layout::ColumnMajor, m as i32, n as i32, 1.0, a, 1, b, 1, ab, m as i32);
    }
}
}} // end macro_rules

impl_cholesky!(f64, dger);
impl_cholesky!(f32, sger);
