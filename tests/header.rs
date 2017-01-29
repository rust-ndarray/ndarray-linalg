
extern crate ndarray;
extern crate ndarray_rand;
extern crate ndarray_linalg;
extern crate ndarray_numtest;
extern crate num_traits;

#[allow(unused_imports)]
use ndarray::prelude::*;
#[allow(unused_imports)]
use ndarray_linalg::prelude::*;
#[allow(unused_imports)]
use ndarray_linalg::util::*;
#[allow(unused_imports)]
use ndarray_numtest::prelude::*;
#[allow(unused_imports)]
use ndarray_rand::RandomExt;
#[allow(unused_imports)]
use num_traits::Float;

pub fn random_hermite(n: usize) -> Array<f64, Ix2> {
    let r_dist = RealNormal::new(0., 1.);
    let a = Array::<f64, _>::random((n, n), r_dist);
    a.dot(&a.t())
}
