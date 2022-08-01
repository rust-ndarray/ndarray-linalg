use ndarray::*;
use ndarray_linalg::{krylov::*, *};

#[test]
fn gmres_real() {
    let m = 28;
    let a: Array2<f64> = random((m, m));
    let b: Array1<f64> = random(m);
    let x0: Array1<f64> = Array1::zeros(m);
    let maxiter = b.len();
    let (x, _) = gmres_mgs(&a, &b, x0, maxiter, 1e-8, 1e-8).unwrap();
    assert_close_l2!(&a.dot(&x), &b, 1e-8);
}

#[test]
fn gmres_complex() {
    let m = 28;
    let a: Array2<c64> = random((m, m));
    let b: Array1<c64> = random(m);
    let x0: Array1<c64> = Array1::zeros(b.len());
    let maxiter = b.len();
    let (x, _) = gmres_mgs(&a, &b, x0, maxiter, 1e-8, 1e-8).unwrap();
    assert_close_l2!(&a.dot(&x), &b, 1e-8);
}
