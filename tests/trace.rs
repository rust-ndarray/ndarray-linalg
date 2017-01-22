
extern crate ndarray;
extern crate ndarray_rand;
extern crate ndarray_linalg;
extern crate ndarray_numtest;

use ndarray::prelude::*;
use ndarray_linalg::prelude::*;
use ndarray_numtest::prelude::*;
use ndarray_rand::RandomExt;

#[test]
fn trace() {
    let r_dist = RealNormal::new(0., 1.);
    let a = Array::<f64, _>::random((3, 3), r_dist);
    a.trace().unwrap().assert_close(a[(0, 0)] + a[(1, 1)] + a[(2, 2)], 1e-7);
}
