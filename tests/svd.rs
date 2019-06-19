use ndarray::*;
use ndarray_linalg::*;
use std::cmp::min;

fn test(a: &Array2<f64>, n: usize, m: usize) {
    test_both(a, n, m);
    test_u(a, n, m);
    test_vt(a, n, m);
}

fn test_both(a: &Array2<f64>, n: usize, m: usize) {
    let answer = a.clone();
    println!("a = \n{:?}", a);
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

fn test_u(a: &Array2<f64>, n: usize, _m: usize) {
    println!("a = \n{:?}", a);
    let (u, _s, vt): (_, Array1<_>, _) = a.svd(true, false).unwrap();
    assert!(u.is_some());
    assert!(vt.is_none());
    let u = u.unwrap();
    assert_eq!(u.dim().0, n);
    assert_eq!(u.dim().1, n);
}

fn test_vt(a: &Array2<f64>, _n: usize, m: usize) {
    println!("a = \n{:?}", a);
    let (u, _s, vt): (_, Array1<_>, _) = a.svd(false, true).unwrap();
    assert!(u.is_none());
    assert!(vt.is_some());
    let vt = vt.unwrap();
    assert_eq!(vt.dim().0, m);
    assert_eq!(vt.dim().1, m);
}

#[test]
fn svd_square() {
    let a = random((3, 3));
    test(&a, 3, 3);
}

#[test]
fn svd_square_t() {
    let a = random((3, 3).f());
    test(&a, 3, 3);
}

#[test]
fn svd_3x4() {
    let a = random((3, 4));
    test(&a, 3, 4);
}

#[test]
fn svd_3x4_t() {
    let a = random((3, 4).f());
    test(&a, 3, 4);
}

#[test]
fn svd_4x3() {
    let a = random((4, 3));
    test(&a, 4, 3);
}

#[test]
fn svd_4x3_t() {
    let a = random((4, 3).f());
    test(&a, 4, 3);
}
