
extern crate rand_extra;
extern crate ndarray;
extern crate ndarray_rand;
#[macro_use]
extern crate ndarray_linalg;

use rand_extra::*;
use ndarray::*;
use ndarray_rand::RandomExt;
use ndarray_linalg::prelude::*;

pub fn random_hermite(n: usize) -> Array<f64, Ix2> {
    let r_dist = RealNormal::new(0., 1.);
    let a = Array::<f64, _>::random((n, n), r_dist);
    a.dot(&a.t())
}

#[test]
fn cholesky() {
    let a = random_hermite(3);
    println!("a = \n{:?}", a);
    let c: Array2<_> = (&a).cholesky(UPLO::Upper).unwrap();
    println!("c = \n{:?}", c);
    println!("cc = \n{:?}", c.t().dot(&c));
    assert_close_l2!(&c.t().dot(&c), &a, 1e-7);
}

#[test]
fn cholesky_t() {
    let a = random_hermite(3);
    println!("a = \n{:?}", a);
    let c: Array2<_> = (&a).cholesky(UPLO::Upper).unwrap();
    println!("c = \n{:?}", c);
    println!("cc = \n{:?}", c.t().dot(&c));
    assert_close_l2!(&c.t().dot(&c), &a, 1e-7);
}
