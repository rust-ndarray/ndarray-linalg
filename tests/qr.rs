
extern crate ndarray;
extern crate ndarray_linalg as linalg;

use ndarray::prelude::*;
use linalg::Matrix;

fn all_close(a: Array<f64, (Ix, Ix)>, b: Array<f64, (Ix, Ix)>) {
    if !a.all_close(&b, 1.0e-7) {
        panic!("\nTwo matrices are not equal:\na = \n{:?}\nb = \n{:?}\n",
               a,
               b);
    }
}

#[test]
fn test_qr_square() {
    let a = arr2(&[[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]);
    let (q, r) = a.clone().qr().unwrap();
    all_close(a, q.dot(&r));
}

#[test]
fn test_qr_3x4() {
    let a = arr2(&[[3.0, 1.0, 1.0, 1.0], [1.0, 3.0, 1.0, 1.0], [1.0, 1.0, 3.0, 1.0]]);
    let (q, r) = a.clone().qr().unwrap();
    all_close(a, q.dot(&r));
}

#[test]
fn test_qr_4x3() {
    let a = arr2(&[[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0], [1.0, 1.0, 1.0]]);
    let (q, r) = a.clone().qr().unwrap();
    all_close(a, q.dot(&r));
}
