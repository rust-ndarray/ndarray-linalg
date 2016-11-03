
extern crate rand;
extern crate ndarray;
extern crate ndarray_rand;
extern crate ndarray_linalg as linalg;

use ndarray::prelude::*;
use linalg::*;
use rand::distributions::*;
use ndarray_rand::RandomExt;

pub fn is_close(a: f64, b: f64) {
    let rel_dev = (a - b).abs() / (a.abs() + b.abs());
    if rel_dev > 1.0e-7 {
        panic!("a={:?}, b={:?} are not almost equal", a, b);
    }
}

pub fn all_close(a: Array<f64, (Ix, Ix)>, b: Array<f64, (Ix, Ix)>) {
    if !a.all_close(&b, 1.0e-7) {
        panic!("\nTwo matrices are not equal:\na = \n{:?}\nb = \n{:?}\n",
               a,
               b);
    }
}

pub fn random_square(n: usize) -> Array<f64, (Ix, Ix)> {
    let r_dist = Range::new(0., 1.);
    Array::<f64, _>::random((n, n), r_dist)
}

pub fn random_unitary(n: usize) -> Array<f64, (Ix, Ix)> {
    let r_dist = Range::new(0., 1.);
    let a = Array::<f64, _>::random((n, n), r_dist);
    let h = a.dot(&a.t()); // hermite
    let (_, u) = h.eigh().unwrap();
    u
}

pub fn random_upper(n: usize, m: usize) -> Array<f64, (Ix, Ix)> {
    let r_dist = Range::new(0., 1.);
    let mut a = Array::<f64, _>::random((n, m), r_dist);

    for ((i, j), val) in a.indexed_iter_mut() {
        if i > j {
            *val = 0.0;
        }
    }
    a
}

#[test]
fn is_unitary() {
    let q = random_unitary(3);
    let i = Array::<f64, _>::eye(3);
    all_close(q.dot(&q.t()), i);
}
