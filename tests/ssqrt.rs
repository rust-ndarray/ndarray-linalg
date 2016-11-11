
extern crate rand;
extern crate ndarray;
extern crate ndarray_rand;
extern crate ndarray_linalg;

use rand::distributions::*;
use ndarray::prelude::*;
use ndarray_linalg::prelude::*;
use ndarray_rand::RandomExt;

fn all_close(a: &Array<f64, Ix2>, b: &Array<f64, Ix2>) {
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
fn ssqrt_symmetric_random_t() {
    let r_dist = Range::new(0., 1.);
    let mut a = Array::<f64, _>::random((3, 3), r_dist);
    a = a.dot(&a.t()).reversed_axes();
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

#[test]
fn ssqrt_sqrt_random_t() {
    let r_dist = Range::new(0., 1.);
    let mut a = Array::<f64, _>::random((3, 3), r_dist);
    a = a.dot(&a.t()).reversed_axes();
    let ar = a.clone().ssqrt().unwrap();
    all_close(&ar.dot(&ar), &a);
}
