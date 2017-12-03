
extern crate ndarray;
#[macro_use]
extern crate ndarray_linalg;
extern crate num_traits;
#[cfg(feature = "lapack-src")]
extern crate lapack_src;

use ndarray::*;
use ndarray_linalg::*;

#[test]
fn n_columns() {
    let a: Array2<f64> = random((3, 2));
    let (n, v) = normalize(a.clone(), NormalizeAxis::Column);
    assert_close_l2!(&n.dot(&from_diag(&v)), &a, 1e-7);
}

#[test]
fn n_rows() {
    let a: Array2<f64> = random((3, 2));
    let (n, v) = normalize(a.clone(), NormalizeAxis::Row);
    assert_close_l2!(&from_diag(&v).dot(&n), &a, 1e-7);
}
