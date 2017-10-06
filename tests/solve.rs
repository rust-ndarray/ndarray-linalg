extern crate ndarray;
#[macro_use]
extern crate ndarray_linalg;
extern crate num_traits;

use ndarray::*;
use ndarray_linalg::*;

#[test]
fn solve_random() {
    let a: Array2<f64> = random((3, 3));
    let x: Array1<f64> = random(3);
    let b = a.dot(&x);
    let y = a.solve_into(b).unwrap();
    assert_close_l2!(&x, &y, 1e-7);
}

#[test]
fn solve_random_t() {
    let a: Array2<f64> = random((3, 3).f());
    let x: Array1<f64> = random(3);
    let b = a.dot(&x);
    let y = a.solve_into(b).unwrap();
    assert_close_l2!(&x, &y, 1e-7);
}
