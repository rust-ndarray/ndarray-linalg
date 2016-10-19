
extern crate rand;
extern crate ndarray;
extern crate ndarray_rand;
extern crate ndarray_linalg as linalg;

use ndarray::prelude::*;
use linalg::SquareMatrix;
use rand::distributions::*;
use ndarray_rand::RandomExt;

fn assert_almost_eq(a: f64, b: f64) {
    let rel_dev = (a - b).abs() / (a.abs() + b.abs());
    if rel_dev > 1.0e-7 {
        panic!("a={:?}, b={:?} are not almost equal", a, b);
    }
}

#[test]
fn trace() {
    let r_dist = Range::new(0., 1.);
    let a = Array::<f64, _>::random((3, 3), r_dist);
    assert_almost_eq(a.trace().unwrap(), a[(0, 0)] + a[(1, 1)] + a[(2, 2)]);
}
