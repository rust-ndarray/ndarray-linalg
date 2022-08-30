use ndarray::*;
use ndarray_linalg::*;

fn eig() {
    let a = arr2(&[[2.0, 1.0, 2.0], [-2.0, 2.0, 1.0], [1.0, 2.0, -2.0]]);
    let (e, vecs) = a.eig().unwrap();
    println!("eigenvalues = \n{:?}", e);
    println!("V = \n{:?}", vecs);
    let a_c: Array2<c64> = a.map(|f| c64::new(*f, 0.0));
    let av = a_c.dot(&vecs);
    println!("AV = \n{:?}", av);
}

fn eigg_real() {
    let a = arr2(&[[1.0 / 2.0.sqrt(), 0.0], [0.0, 1.0]]);
    let b = arr2(&[[0.0, 1.0], [-1.0 / 2.0.sqrt(), 0.0]]);
    let (e, vecs) = a.clone().eigg(&b).unwrap();
    println!("eigenvalues = \n{:?}", e);
    println!("V = \n{:?}", vecs);
}

fn eigg_complex() {
    let a = arr2(&[
        [c64::complex(-3.84, 2.25), c64::complex(-3.84, 2.25)],
        [c64::complex(-3.84, 2.25), c64::complex(-3.84, 2.25)],
    ]);
    let b = a.clone();
    let (e, vecs) = a.clone().eigg(&b).unwrap();
    println!("eigenvalues = \n{:?}", &e);
    println!("V = \n{:?}", vecs);
}

fn main() {
    eig();
    eigg_real();
    eigg_complex();
}
