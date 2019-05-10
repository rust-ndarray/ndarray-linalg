//! Assertions for array

use ndarray::*;
use std::fmt::Debug;

use super::norm::*;
use super::types::*;

/// check two values are close in terms of the relative torrence
pub fn rclose<A: Scalar>(test: A, truth: A, rtol: A::Real) {
    let dev = (test - truth).abs() / truth.abs();
    if dev > rtol {
        eprintln!("Expected = {}", truth);
        eprintln!("Actual   = {}", test);
        panic!("Too large deviation in relative torrence: {}", dev);
    }
}

/// check two values are close in terms of the absolute torrence
pub fn aclose<A: Scalar>(test: A, truth: A, atol: A::Real) {
    let dev = (test - truth).abs();
    if dev > atol {
        eprintln!("Expected = {}", truth);
        eprintln!("Actual   = {}", test);
        panic!("Too large deviation in absolute torrence: {}", dev);
    }
}

/// check two arrays are close in maximum norm
pub fn close_max<A, S1, S2, D>(test: &ArrayBase<S1, D>, truth: &ArrayBase<S2, D>, atol: A::Real)
where
    A: Scalar + Lapack,
    S1: Data<Elem = A>,
    S2: Data<Elem = A>,
    D: Dimension,
    D::Pattern: PartialEq + Debug,
{
    let tol = (test - truth).norm_max();
    if tol > atol {
        eprintln!("Expected:\n{}", truth);
        eprintln!("Actual:\n{}", test);
        panic!("Too large deviation in maximum norm: {}", tol);
    }
}

/// check two arrays are close in L1 norm
pub fn close_l1<A, S1, S2, D>(test: &ArrayBase<S1, D>, truth: &ArrayBase<S2, D>, rtol: A::Real)
where
    A: Scalar + Lapack,
    S1: Data<Elem = A>,
    S2: Data<Elem = A>,
    D: Dimension,
    D::Pattern: PartialEq + Debug,
{
    let tol = (test - truth).norm_l1() / truth.norm_l1();
    if tol > rtol {
        eprintln!("Expected:\n{}", truth);
        eprintln!("Actual:\n{}", test);
        panic!("Too large deviation in L1-norm: {}", tol);
    }
}

/// check two arrays are close in L2 norm
pub fn close_l2<A, S1, S2, D>(test: &ArrayBase<S1, D>, truth: &ArrayBase<S2, D>, rtol: A::Real)
where
    A: Scalar + Lapack,
    S1: Data<Elem = A>,
    S2: Data<Elem = A>,
    D: Dimension,
    D::Pattern: PartialEq + Debug,
{
    let tol = (test - truth).norm_l2() / truth.norm_l2();
    if tol > rtol {
        eprintln!("Expected:\n{}", truth);
        eprintln!("Actual:\n{}", test);
        panic!("Too large deviation in L2-norm: {}", tol);
    }
}

macro_rules! generate_assert {
    ($assert:ident, $close:path) => {
        #[macro_export]
        macro_rules! $assert {
            ($test: expr,$truth: expr,$tol: expr) => {
                $crate::$close($test, $truth, $tol);
            };
        }
    };
} // generate_assert!

generate_assert!(assert_rclose, rclose);
generate_assert!(assert_aclose, aclose);
generate_assert!(assert_close_max, close_max);
generate_assert!(assert_close_l1, close_l1);
generate_assert!(assert_close_l2, close_l2);
