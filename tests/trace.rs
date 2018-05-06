extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

#[test]
fn trace() {
    let a: Array2<f64> = random((3, 3));
    assert_rclose!(a.trace().unwrap(), a[(0, 0)] + a[(1, 1)] + a[(2, 2)], 1e-7);
}
