
extern crate ndarray;
extern crate ndarray_linalg as linalg;

use ndarray::prelude::*;
use linalg::Matrix;

fn test_square() {
    println!("\n=== Test QR for square matrix ===");
    let a = arr2(&[[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]);
    println!("a = \n{:?}", &a);
    let (q, r) = a.clone().qr().unwrap();
    println!("Q = \n{:?}", &q);
    println!("R = \n{:?}", &r);
}

fn test_3x4() {
    println!("\n=== Test QR for 3x4 matrix ===");
    let a = arr2(&[[3.0, 1.0, 1.0, 1.0], [1.0, 3.0, 1.0, 1.0], [1.0, 1.0, 3.0, 1.0]]);
    println!("a = \n{:?}", &a);
    let (q, r) = a.clone().qr().unwrap();
    println!("Q = \n{:?}", &q);
    println!("R = \n{:?}", &r);
}

fn test_4x3() {
    println!("\n=== Test QR for 4x3 matrix ===");
    let a = arr2(&[[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0], [1.0, 1.0, 1.0]]);
    println!("a = \n{:?}", &a);
    let (q, r) = a.clone().qr().unwrap();
    println!("Q = \n{:?}", &q);
    println!("R = \n{:?}", &r);
}

fn main() {
    test_square();
    test_3x4();
    test_4x3();
}
