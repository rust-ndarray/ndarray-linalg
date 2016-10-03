
extern crate ndarray;
extern crate ndarray_linalg as linalg;

use ndarray::prelude::*;
use linalg::SquareMatrix;

fn main() {
    let a2 = arr2(&[[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]);
    let (e, vecs) = a2.eig();
    println!("eigenvalues = {:?}", e);
    println!("eigenvectors = \n{:?}", vecs);
}
