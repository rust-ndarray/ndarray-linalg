//! Assertions for array

use std::iter::Sum;
use num_traits::Float;
use ndarray::*;

use super::types::*;
use super::vector::*;

pub fn rclose<A, Tol>(test: A, truth: A, rtol: Tol) -> Result<Tol, Tol>
    where A: LinalgScalar + Absolute<Output = Tol>,
          Tol: Float
{
    let dev = (test - truth).abs() / truth.abs();
    if dev < rtol { Ok(dev) } else { Err(dev) }
}

pub fn aclose<A, Tol>(test: A, truth: A, atol: Tol) -> Result<Tol, Tol>
    where A: LinalgScalar + Absolute<Output = Tol>,
          Tol: Float
{
    let dev = (test - truth).abs();
    if dev < atol { Ok(dev) } else { Err(dev) }
}

/// check two arrays are close in maximum norm
pub fn close_max<A, Tol, S1, S2, D>(test: &ArrayBase<S1, D>, truth: &ArrayBase<S2, D>, atol: Tol) -> Result<Tol, Tol>
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
pub fn close_l1<A, Tol, S1, S2, D>(test: &ArrayBase<S1, D>, truth: &ArrayBase<S2, D>, rtol: Tol) -> Result<Tol, Tol>
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
pub fn close_l2<A, Tol, S1, S2, D>(test: &ArrayBase<S1, D>, truth: &ArrayBase<S2, D>, rtol: Tol) -> Result<Tol, Tol>
    where A: LinalgScalar + Absolute<Output = Tol>,
          Tol: Float + Sum,
          S1: Data<Elem = A>,
          S2: Data<Elem = A>,
          D: Dimension
{
    let tol = (test - truth).norm_l2() / truth.norm_l2();
    if tol < rtol { Ok(tol) } else { Err(tol) }
}

macro_rules! generate_assert {
    ($assert:ident, $close:path) => {
#[macro_export]
macro_rules! $assert {
    ($test:expr, $truth:expr, $tol:expr) => {
        $close($test, $truth, $tol).unwrap();
    };
    ($test:expr, $truth:expr, $tol:expr; $comment:expr) => {
        $close($test, $truth, $tol).expect($comment);
    };
}
}} // generate_assert!

generate_assert!(assert_rclose, rclose);
generate_assert!(assert_aclose, aclose);
generate_assert!(assert_close_max, close_max);
generate_assert!(assert_close_l1, close_l1);
generate_assert!(assert_close_l2, close_l2);
