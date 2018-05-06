extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

fn main() {
    let a = arr2(&[[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]);
    let (e, vecs): (Array1<_>, Array2<_>) = a.clone().eigh(UPLO::Upper).unwrap();
    println!("eigenvalues = \n{:?}", e);
    println!("V = \n{:?}", vecs);
    let av = a.dot(&vecs);
    println!("AV = \n{:?}", av);
}
