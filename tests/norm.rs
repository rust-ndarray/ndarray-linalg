
extern crate ndarray;
extern crate ndarray_linalg as linalg;
extern crate num_traits;

use ndarray::prelude::*;
use linalg::{Matrix, Vector};
use num_traits::float::Float;

fn assert_almost_eq(a: f64, b: f64) {
    let rel_dev = (a - b).abs() / (a.abs() + b.abs());
    if rel_dev > 1.0e-7 {
        panic!("a={:?}, b={:?} are not almost equal", a, b);
    }
}

#[test]
fn vector_norm() {
    let a = Array::range(1., 10., 1.);
    assert_almost_eq(a.norm(), 285.0.sqrt());
}

#[test]
fn matrix_norm_square() {
    let a = Array::range(1., 10., 1.).into_shape((3, 3)).unwrap();
    assert_almost_eq(a.norm_1(), 18.0);
    assert_almost_eq(a.norm_i(), 24.0);
    assert_almost_eq(a.norm_f(), 285.0.sqrt());
}

#[test]
fn matrix_norm_square_t() {
    let a = Array::range(1., 10., 1.).into_shape((3, 3)).unwrap().reversed_axes();
    assert_almost_eq(a.norm_1(), 24.0);
    assert_almost_eq(a.norm_i(), 18.0);
    assert_almost_eq(a.norm_f(), 285.0.sqrt());
}

#[test]
fn matrix_norm_3x4() {
    let a = Array::range(1., 13., 1.).into_shape((3, 4)).unwrap();
    assert_almost_eq(a.norm_1(), 24.0);
    assert_almost_eq(a.norm_i(), 42.0);
    assert_almost_eq(a.norm_f(), 650.0.sqrt());
}

#[test]
fn matrix_norm_3x4_t() {
    let a = Array::range(1., 13., 1.)
        .into_shape((3, 4))
        .unwrap()
        .reversed_axes();
    assert_almost_eq(a.norm_1(), 42.0);
    assert_almost_eq(a.norm_i(), 24.0);
    assert_almost_eq(a.norm_f(), 650.0.sqrt());
}

#[test]
fn matrix_norm_4x3() {
    let a = Array::range(1., 13., 1.).into_shape((4, 3)).unwrap();
    assert_almost_eq(a.norm_1(), 30.0);
    assert_almost_eq(a.norm_i(), 33.0);
    assert_almost_eq(a.norm_f(), 650.0.sqrt());
}

#[test]
fn matrix_norm_4x3_t() {
    let a = Array::range(1., 13., 1.)
        .into_shape((4, 3))
        .unwrap()
        .reversed_axes();
    assert_almost_eq(a.norm_1(), 33.0);
    assert_almost_eq(a.norm_i(), 30.0);
    assert_almost_eq(a.norm_f(), 650.0.sqrt());
}
