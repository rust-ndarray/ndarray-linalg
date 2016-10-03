
extern crate ndarray;
extern crate lapack;

use ndarray::prelude::*;
use lapack::fortran::*;

fn eigs_(n: usize, mut a: Vec<f64>) -> (Vec<f64>, Vec<f64>) {
    let mut w = vec![0.0; n as usize];
    let mut work = vec![0.0; 4 * n as usize];
    let lwork = 4 * n;
    let mut info = 0;
    dsyev(b'V',
          b'U',
          n as i32,
          &mut a,
          n as i32,
          &mut w,
          &mut work,
          lwork as i32,
          &mut info);
    assert_eq!(info, 0);
    (w, a)
}

pub fn eigs(a: Array<f64, (Ix, Ix)>) -> (Array<f64, Ix>, Array<f64, (Ix, Ix)>) {
    let rows = a.rows();
    let cols = a.cols();
    assert_eq!(rows, cols);
    let (e, vecs) = eigs_(rows, a.into_raw_vec());
    let ea = Array::from_vec(e);
    let va = Array::from_vec(vecs).into_shape((rows, cols)).unwrap();
    (ea, va)
}
