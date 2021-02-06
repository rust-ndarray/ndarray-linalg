use ndarray::*;
use ndarray_linalg::*;

#[test]
fn vector_norm() {
    let a = Array::range(1., 10., 1.);
    assert_rclose!(a.norm(), 285.0.sqrt(), 1e-7);
}

#[test]
fn vector_norm_l1() {
    let a = arr1(&[1.0, -1.0]);
    assert_rclose!(a.norm_l1(), 2.0, 1e-7);
    let b = arr2(&[[0.0, -1.0], [1.0, 0.0]]);
    assert_rclose!(b.norm_l1(), 2.0, 1e-7);
}

#[test]
fn vector_norm_max() {
    let a = arr1(&[1.0, 1.0, -3.0]);
    assert_rclose!(a.norm_max(), 3.0, 1e-7);
    let b = arr2(&[[1.0, 3.0], [1.0, -4.0]]);
    assert_rclose!(b.norm_max(), 4.0, 1e-7);
}

#[test]
fn vector_norm_l1_rc() {
    let a = rcarr1(&[1.0, -1.0]);
    assert_rclose!(a.norm_l1(), 2.0, 1e-7);
    let b = rcarr2(&[[0.0, -1.0], [1.0, 0.0]]);
    assert_rclose!(b.norm_l1(), 2.0, 1e-7);
}

#[test]
fn vector_norm_max_rc() {
    let a = rcarr1(&[1.0, 1.0, -3.0]);
    assert_rclose!(a.norm_max(), 3.0, 1e-7);
    let b = rcarr2(&[[1.0, 3.0], [1.0, -4.0]]);
    assert_rclose!(b.norm_max(), 4.0, 1e-7);
}
