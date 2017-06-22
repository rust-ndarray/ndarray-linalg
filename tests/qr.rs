
extern crate ndarray;
#[macro_use]
extern crate ndarray_linalg;

use std::cmp::min;
use ndarray::*;
use ndarray_linalg::*;

fn test(a: Array2<f64>, n: usize, m: usize) {
    let ans = a.clone();
    println!("a = \n{:?}", &a);
    let (q, r): (Array2<_>, Array2<_>) = a.qr().unwrap();
    println!("q = \n{:?}", &q);
    println!("r = \n{:?}", &r);
    assert_close_l2!(&q.t().dot(&q), &Array::eye(min(n, m)), 1e-7);
    assert_close_l2!(&q.dot(&r), &ans, 1e-7);
    assert_close_l2!(&r.clone().into_triangular(UPLO::Upper), &r, 1e-7);
}

#[test]
fn qr_square() {
    let a = random((3, 3));
    test(a, 3, 3);
}

#[test]
fn qr_square_t() {
    let a = random((3, 3).f());
    test(a, 3, 3);
}

#[test]
fn qr_3x4() {
    let a = random((3, 4));
    test(a, 3, 4);
}

#[test]
fn qr_3x4_t() {
    let a = random((3, 4).f());
    test(a, 3, 4);
}

#[test]
fn qr_4x3() {
    let a = random((4, 3));
    test(a, 4, 3);
}

#[test]
fn qr_4x3_t() {
    let a = random((4, 3).f());
    test(a, 4, 3);
}
