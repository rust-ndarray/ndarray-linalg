//! Assertions for array

use std::iter::Sum;
use num_traits::Float;
use ndarray::*;

use super::vector::*;

pub trait Close: Absolute {
    fn rclose(self, truth: Self, relative_tol: Self::Output) -> Result<Self::Output, Self::Output>;
    fn aclose(self, truth: Self, absolute_tol: Self::Output) -> Result<Self::Output, Self::Output>;
}

macro_rules! impl_AssertClose {
    ($scalar:ty) => {
impl Close for $scalar {
    fn rclose(self, truth: Self, rtol: Self::Output) -> Result<Self::Output, Self::Output> {
        let dev = (self - truth).abs() / truth.abs();
        if dev < rtol {
            Ok(dev)
        } else {
            Err(dev)
        }
    }

    fn aclose(self, truth: Self, atol: Self::Output) -> Result<Self::Output, Self::Output> {
        let dev = (self - truth).abs();
        if dev < atol {
            Ok(dev)
        } else {
            Err(dev)
        }
    }
}
}} // impl_AssertClose
impl_AssertClose!(f64);
impl_AssertClose!(f32);

#[macro_export]
macro_rules! assert_rclose {
    ($test:expr, $truth:expr, $tol:expr) => {
        $test.rclose($truth, $tol).unwrap();
    };
    ($test:expr, $truth:expr, $tol:expr; $comment:expr) => {
        $test.rclose($truth, $tol).expect($comment);
    };
}

#[macro_export]
macro_rules! assert_aclose {
    ($test:expr, $truth:expr, $tol:expr) => {
        $test.aclose($truth, $tol).unwrap();
    };
    ($test:expr, $truth:expr, $tol:expr; $comment:expr) => {
        $test.aclose($truth, $tol).expect($comment);
    };
}

/// check two arrays are close in maximum norm
pub fn all_close_max<A, Tol, S1, S2, D>(test: &ArrayBase<S1, D>,
                                        truth: &ArrayBase<S2, D>,
                                        atol: Tol)
                                        -> Result<Tol, Tol>
    where A: LinalgScalar + Absolute<Output = Tol>,
          Tol: Float + Sum,
          S1: Data<Elem = A>,
          S2: Data<Elem = A>,
          D: Dimension
{
    let tol = (test - truth).norm_max();
    if tol < atol { Ok(tol) } else { Err(tol) }
}

/// check two arrays are close in L1 norm
pub fn all_close_l1<A, Tol, S1, S2, D>(test: &ArrayBase<S1, D>, truth: &ArrayBase<S2, D>, rtol: Tol) -> Result<Tol, Tol>
    where A: LinalgScalar + Absolute<Output = Tol>,
          Tol: Float + Sum,
          S1: Data<Elem = A>,
          S2: Data<Elem = A>,
          D: Dimension
{
    let tol = (test - truth).norm_l1() / truth.norm_l1();
    if tol < rtol { Ok(tol) } else { Err(tol) }
}

/// check two arrays are close in L2 norm
pub fn all_close_l2<A, Tol, S1, S2, D>(test: &ArrayBase<S1, D>, truth: &ArrayBase<S2, D>, rtol: Tol) -> Result<Tol, Tol>
    where A: LinalgScalar + Absolute<Output = Tol>,
          Tol: Float + Sum,
          S1: Data<Elem = A>,
          S2: Data<Elem = A>,
          D: Dimension
{
    let tol = (test - truth).norm_l2() / truth.norm_l2();
    if tol < rtol { Ok(tol) } else { Err(tol) }
}
