
extern crate ndarray;
extern crate lapack;

use ndarray::prelude::*;
use lapack::fortran::*;

/// eigenvalue decompostion for symmetric matrix
/// (use upper matrix)
fn eigs(n: usize, mut a: Vec<f64>) -> (Vec<f64>, Vec<f64>) {
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

fn eigs_wrap(a: Array<f64, (Ix, Ix)>) -> (Array<f64, Ix>, Array<f64, (Ix, Ix)>) {
    let rows = a.rows();
    let cols = a.cols();
    assert_eq!(rows, cols);
    let (e, vecs) = eigs(rows, a.into_raw_vec());
    let ea = Array::from_vec(e);
    let va = Array::from_vec(vecs).into_shape((rows, cols)).unwrap();
    (ea, va)
}

fn main() {
    let n = 3;
    let a = vec![3.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 3.0];
    let (w, _) = eigs(n as usize, a);
    for (one, another) in w.iter().zip(&[2.0, 2.0, 5.0]) {
        assert!((one - another).abs() < 1e-14);
    }
    let a2 = arr2(&[[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]);
    let (e, vecs) = eigs_wrap(a2);
    println!("eigenvalues = {:?}", e);
    println!("eigenvectors = \n{:?}", vecs);
}
