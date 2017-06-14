
extern crate ndarray;
#[macro_use]
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::prelude::*;

#[test]
fn cholesky() {
    let a: Array2<f64> = random_hpd(3);
    println!("a = \n{:?}", a);
    let c: Array2<_> = (&a).cholesky(UPLO::Upper).unwrap();
    println!("c = \n{:?}", c);
    println!("cc = \n{:?}", c.t().dot(&c));
    assert_close_l2!(&c.t().dot(&c), &a, 1e-7);
}

#[test]
fn cholesky_t() {
    let a: Array2<f64> = random_hpd(3);
    println!("a = \n{:?}", a);
    let c: Array2<_> = (&a).cholesky(UPLO::Upper).unwrap();
    println!("c = \n{:?}", c);
    println!("cc = \n{:?}", c.t().dot(&c));
    assert_close_l2!(&c.t().dot(&c), &a, 1e-7);
}
