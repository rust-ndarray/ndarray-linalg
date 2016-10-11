
extern crate ndarray;
extern crate ndarray_linalg as linalg;

use ndarray::prelude::*;
use linalg::Matrix;

fn main() {
    let a = arr2(&[[3.0, 1.0, 1.0, 1.0], [1.0, 3.0, 1.0, 1.0], [1.0, 1.0, 3.0, 1.0]]);
    println!("a = \n{:?}", &a);
    let (q, r) = a.clone().qr().unwrap();
    println!("Q = \n{:?}", &q);
    println!("R = \n{:?}", &r);
}
