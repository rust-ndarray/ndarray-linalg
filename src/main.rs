
extern crate ndarray;
extern crate ndarray_linalg as linalg;

use ndarray::prelude::*;

fn main() {
    let a2 = arr2(&[[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]);
    let (e, vecs) = linalg::eigs(a2);
    println!("eigenvalues = {:?}", e);
    println!("eigenvectors = \n{:?}", vecs);
}
