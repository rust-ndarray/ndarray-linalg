
extern crate ndarray;
extern crate ndarray_linalg as linalg;

use ndarray::prelude::*;
use linalg::Matrix;

fn assert_almost_eq(a: f64, b: f64) {
    let rel_dev = (a - b).abs() / (a.abs() + b.abs());
    if rel_dev > 1.0e-7 {
        panic!("a={:?}, b={:?} are not almost equal", a, b);
    }
}

#[test]
fn test_matrix_norm_3x4() {
    let a = arr2(&[[3.0, 1.0, 1.0, 1.0], [1.0, 3.0, 1.0, 1.0], [1.0, 1.0, 3.0, 1.0]]);
    assert_almost_eq(a.norm_1(), 5.0);
    assert_almost_eq(a.norm_i(), 6.0);
}

#[test]
fn test_matrix_norm_3x4_t() {
    let a = arr2(&[[3.0, 1.0, 1.0, 1.0], [1.0, 3.0, 1.0, 1.0], [1.0, 1.0, 3.0, 1.0]])
        .reversed_axes();
    assert_almost_eq(a.norm_1(), 6.0);
    assert_almost_eq(a.norm_i(), 5.0);
}

#[test]
fn test_matrix_norm_4x3() {
    let a = arr2(&[[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0], [1.0, 1.0, 1.0]]);
    assert_almost_eq(a.norm_1(), 6.0);
    assert_almost_eq(a.norm_i(), 5.0);
}

#[test]
fn test_matrix_norm_4x3_t() {
    let a = arr2(&[[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0], [1.0, 1.0, 1.0]])
        .reversed_axes();
    assert_almost_eq(a.norm_1(), 5.0);
    assert_almost_eq(a.norm_i(), 6.0);
}
