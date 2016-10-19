
extern crate rand;
extern crate ndarray;
extern crate ndarray_rand;
extern crate ndarray_linalg as linalg;

use ndarray::prelude::*;
use linalg::SquareMatrix;
use rand::distributions::*;
use ndarray_rand::RandomExt;

fn all_close(a: &Array<f64, (Ix, Ix)>, b: &Array<f64, (Ix, Ix)>) {
    if !a.all_close(b, 1.0e-7) {
        panic!("\nTwo matrices are not equal:\na = \n{:?}\nb = \n{:?}\n",
               a,
               b);
    }
}

#[test]
fn ssqrt_symmetric_random() {
    let r_dist = Range::new(0., 1.);
    let mut a = Array::<f64, _>::random((3, 3), r_dist);
    a = a.dot(&a.t());
    let ar = a.clone().ssqrt().unwrap();
    all_close(&ar.clone().reversed_axes(), &ar);
}

#[test]
fn ssqrt_sqrt_random() {
    let r_dist = Range::new(0., 1.);
    let mut a = Array::<f64, _>::random((3, 3), r_dist);
    a = a.dot(&a.t());
    let ar = a.clone().ssqrt().unwrap();
    all_close(&ar.dot(&ar), &a);
}
