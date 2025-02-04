use ndarray::*;
use ndarray_linalg::*;

#[test]
fn layout_c_3x1() {
    let a: Array2<f64> = Array::zeros((3, 1));
    println!("a = {:?}", &a);
    assert_eq!(a.layout().unwrap(), MatrixLayout::C { row: 3, lda: 1 });
}

#[test]
fn layout_f_3x1() {
    let a: Array2<f64> = Array::zeros((3, 1).f());
    println!("a = {:?}", &a);
    assert_eq!(a.layout().unwrap(), MatrixLayout::F { col: 1, lda: 3 });
}

#[test]
fn layout_c_3x2() {
    let a: Array2<f64> = Array::zeros((3, 2));
    println!("a = {:?}", &a);
    assert_eq!(a.layout().unwrap(), MatrixLayout::C { row: 3, lda: 2 });
}

#[test]
fn layout_f_3x2() {
    let a: Array2<f64> = Array::zeros((3, 2).f());
    println!("a = {:?}", &a);
    assert_eq!(a.layout().unwrap(), MatrixLayout::F { col: 2, lda: 3 });
}
