
extern crate ndarray;
extern crate ndarray_rand;
extern crate ndarray_linalg;
extern crate ndarray_numtest;
extern crate num_traits;

#[allow(unused_imports)]
use ndarray::*;
#[allow(unused_imports)]
use ndarray_linalg::prelude::*;
#[allow(unused_imports)]
use ndarray_numtest::prelude::*;
#[allow(unused_imports)]
use ndarray_rand::RandomExt;
#[allow(unused_imports)]
use num_traits::Float;

pub fn random_owned(n: usize, m: usize, reversed: bool) -> Array<f64, Ix2> {
    let r_dist = RealNormal::new(0., 1.);
    if reversed {
        Array::random((m, n), r_dist).reversed_axes()
    } else {
        Array::random((n, m), r_dist)
    }
}
pub fn random_shared(n: usize, m: usize, reversed: bool) -> RcArray<f64, Ix2> {
    let r_dist = RealNormal::new(0., 1.);
    if reversed {
        RcArray::random((m, n), r_dist).reversed_axes()
    } else {
        RcArray::random((n, m), r_dist)
    }
}

pub fn random_square(n: usize) -> Array<f64, Ix2> {
    let r_dist = RealNormal::new(0., 1.);
    Array::<f64, _>::random((n, n), r_dist)
}

pub fn random_hermite(n: usize) -> Array<f64, Ix2> {
    let r_dist = RealNormal::new(0., 1.);
    let a = Array::<f64, _>::random((n, n), r_dist);
    a.dot(&a.t())
}
