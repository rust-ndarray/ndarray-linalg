
extern crate ndarray;
extern crate ndarray_linalg as linalg;

use ndarray::prelude::*;
use linalg::SquareMatrix;

fn main() {
    let a = arr2(&[[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]);
    let (e, vecs) = a.eig();
    println!("eigenvalues = {:?}", e);
    println!("eigenvectors = \n{:?}", vecs);
}
