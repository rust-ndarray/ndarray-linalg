
extern crate ndarray;
#[macro_use]
extern crate ndarray_linalg;

use std::cmp::min;
use ndarray::*;
use ndarray_linalg::*;

fn test(a: Array2<f64>, n: usize, m: usize) {
    let answer = a.clone();
    println!("a = \n{:?}", &a);
    let (u, s, vt): (_, Array1<_>, _) = a.svd(true, true).unwrap();
    let u: Array2<_> = u.unwrap();
    let vt: Array2<_> = vt.unwrap();
    println!("u = \n{:?}", &u);
    println!("s = \n{:?}", &s);
    println!("v = \n{:?}", &vt);
    let mut sm = Array::zeros((n, m));
    for i in 0..min(n, m) {
        sm[(i, i)] = s[i];
    }
    assert_close_l2!(&u.dot(&sm).dot(&vt), &answer, 1e-7);
}

#[test]
fn svd_square() {
    let a = random((3, 3));
    test(a, 3, 3);
}

#[test]
fn svd_square_t() {
    let a = random((3, 3).f());
    test(a, 3, 3);
}

#[test]
fn svd_3x4() {
    let a = random((3, 4));
    test(a, 3, 4);
}

#[test]
fn svd_3x4_t() {
    let a = random((3, 4).f());
    test(a, 3, 4);
}

#[test]
fn svd_4x3() {
    let a = random((4, 3));
    test(a, 4, 3);
}

#[test]
fn svd_4x3_t() {
    let a = random((4, 3).f());
    test(a, 4, 3);
}
