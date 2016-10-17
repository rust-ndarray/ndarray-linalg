
extern crate rand;
extern crate ndarray;
extern crate ndarray_rand;
extern crate ndarray_linalg as linalg;

use ndarray::prelude::*;
use linalg::SquareMatrix;
use rand::distributions::*;
use ndarray_rand::RandomExt;

fn all_close(a: Array<f64, (Ix, Ix)>, b: Array<f64, (Ix, Ix)>) {
    if !a.all_close(&b, 1.0e-7) {
        panic!("\nTwo matrices are not equal:\na = \n{:?}\nb = \n{:?}\n",
               a,
               b);
    }
}

#[test]
fn inv_random() {
    let r_dist = Range::new(0., 1.);
    let a = Array::<f64, _>::random((3, 3), r_dist);
    let ai = a.clone().inv().unwrap();
    let id = Array::eye(3);
    all_close(ai.dot(&a), id);
}

#[test]
fn inv_random_t() {
    let r_dist = Range::new(0., 1.);
    let a = Array::<f64, _>::random((3, 3), r_dist).reversed_axes();
    let ai = a.clone().inv().unwrap();
    let id = Array::eye(3);
    all_close(ai.dot(&a), id);
}

#[test]
#[should_panic]
fn inv_error() {
    // do not have inverse
    let a = Array::range(1., 10., 1.).into_shape((3, 3)).unwrap();
    let _ = a.clone().inv().unwrap();
}
