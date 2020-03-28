extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

fn main() {
    let a = arr2(&[[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 3.0, 2.0]]);
    let (e, vecs) = a.clone().eig().unwrap();
    println!("eigenvalues = \n{:?}", e);
    println!("V = \n{:?}", vecs.t());
    let av = a.dot(&vecs.t());
    println!("AV = \n{:?}", av);
}