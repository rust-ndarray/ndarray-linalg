
extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use ndarray_linalg::layout::MatrixLayout;

#[test]
fn layout_c_3x1() {
    let a: Array2<f64> = Array::zeros((3, 1));
    println!("a = {:?}", &a);
    assert_eq!(a.layout().unwrap(), MatrixLayout::C((3, 1)));
}

#[test]
fn layout_f_3x1() {
    let a: Array2<f64> = Array::zeros((3, 1).f());
    println!("a = {:?}", &a);
    assert_eq!(a.layout().unwrap(), MatrixLayout::F((1, 3)));
}

#[test]
fn layout_c_3x2() {
    let a: Array2<f64> = Array::zeros((3, 2));
    println!("a = {:?}", &a);
    assert_eq!(a.layout().unwrap(), MatrixLayout::C((3, 2)));
}

#[test]
fn layout_f_3x2() {
    let a: Array2<f64> = Array::zeros((3, 2).f());
    println!("a = {:?}", &a);
    assert_eq!(a.layout().unwrap(), MatrixLayout::F((2, 3)));
}
