use ndarray::*;
use ndarray_linalg::*;

fn main() {
    let a = arr2(&[[2.0, 1.0, 2.0], [-2.0, 2.0, 1.0], [1.0, 2.0, -2.0]]);
    let (e, vecs) = a.clone().eig().unwrap();
    println!("eigenvalues = \n{:?}", e);
    println!("V = \n{:?}", vecs);
    let a_c: Array2<c64> = a.map(|f| c64::new(*f, 0.0));
    let av = a_c.dot(&vecs);
    println!("AV = \n{:?}", av);
}