
extern crate ndarray;
#[macro_use]
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::prelude::*;

fn test1d<A, Sa, Sb, Tol>(uplo: UPLO, a: ArrayBase<Sa, Ix2>, b: ArrayBase<Sb, Ix1>, tol: Tol)
    where A: Field + Absolute<Output = Tol>,
          Sa: Data<Elem = A>,
          Sb: DataMut<Elem = A> + DataOwned,
          Tol: RealField
{
    println!("a = {:?}", &a);
    println!("b = {:?}", &b);
    let x = a.solve_triangular(uplo, Diag::NonUnit, &b).unwrap();
    println!("x = {:?}", &x);
    let b_ = a.dot(&x);
    println!("Ax = {:?}", &b_);
    assert_close_l2!(&b_, &b, tol);
}

fn test2d<A, Sa, Sb, Tol>(uplo: UPLO, a: ArrayBase<Sa, Ix2>, b: ArrayBase<Sb, Ix2>, tol: Tol)
    where A: Field + Absolute<Output = Tol>,
          Sa: Data<Elem = A>,
          Sb: DataMut<Elem = A> + DataOwned + DataClone,
          Tol: RealField
{
    println!("a = {:?}", &a);
    println!("b = {:?}", &b);
    let ans = b.clone();
    let x = a.solve_triangular(uplo, Diag::NonUnit, b).unwrap();
    println!("x = {:?}", &x);
    let b_ = a.dot(&x);
    println!("Ax = {:?}", &b_);
    assert_close_l2!(&b_, &ans, tol);
}

#[test]
fn triangular_1d_upper() {
    let n = 3;
    let b: Array1<f64> = random_vector(n);
    let a: Array2<f64> = random_square(n).into_triangular(UPLO::Upper);
    test1d(UPLO::Upper, a, b, 1e-7);
}

#[test]
fn triangular_1d_lower() {
    let n = 3;
    let b: Array1<f64> = random_vector(n);
    let a: Array2<f64> = random_square(n).into_triangular(UPLO::Lower);
    test1d(UPLO::Lower, a, b, 1e-7);
}

#[test]
fn triangular_1d_lower_t() {
    let n = 3;
    let b: Array1<f64> = random_vector(n);
    let a: Array2<f64> = random_square(n).into_triangular(UPLO::Lower).reversed_axes();
    test1d(UPLO::Upper, a, b, 1e-7);
}

#[test]
fn triangular_1d_upper_t() {
    let n = 3;
    let b: Array1<f64> = random_vector(n);
    let a: Array2<f64> = random_square(n).into_triangular(UPLO::Upper).reversed_axes();
    test1d(UPLO::Lower, a, b, 1e-7);
}

#[test]
fn triangular_2d_upper() {
    let b: Array2<f64> = random_matrix(3, 4);
    let a: Array2<f64> = random_square(3).into_triangular(UPLO::Upper);
    test2d(UPLO::Upper, a, b, 1e-7);
}

#[test]
fn triangular_2d_lower() {
    let b: Array2<f64> = random_matrix(3, 4);
    let a: Array2<f64> = random_square(3).into_triangular(UPLO::Lower);
    test2d(UPLO::Lower, a, b, 1e-7);
}

#[test]
fn triangular_2d_lower_t() {
    let b: Array2<f64> = random_matrix(3, 4);
    let a: Array2<f64> = random_square(3).into_triangular(UPLO::Lower).reversed_axes();
    test2d(UPLO::Upper, a, b, 1e-7);
}

#[test]
fn triangular_2d_upper_t() {
    let b: Array2<f64> = random_matrix(3, 4);
    let a: Array2<f64> = random_square(3).into_triangular(UPLO::Upper).reversed_axes();
    test2d(UPLO::Lower, a, b, 1e-7);
}

#[test]
fn triangular_2d_upper_bt() {
    let b: Array2<f64> = random_matrix(4, 3).reversed_axes();
    let a: Array2<f64> = random_square(3).into_triangular(UPLO::Upper);
    test2d(UPLO::Upper, a, b, 1e-7);
}

#[test]
fn triangular_2d_lower_bt() {
    let b: Array2<f64> = random_matrix(4, 3).reversed_axes();
    let a: Array2<f64> = random_square(3).into_triangular(UPLO::Lower);
    test2d(UPLO::Lower, a, b, 1e-7);
}

#[test]
fn triangular_2d_lower_t_bt() {
    let b: Array2<f64> = random_matrix(4, 3).reversed_axes();
    let a: Array2<f64> = random_square(3).into_triangular(UPLO::Lower).reversed_axes();
    test2d(UPLO::Upper, a, b, 1e-7);
}

#[test]
fn triangular_2d_upper_t_bt() {
    let b: Array2<f64> = random_matrix(4, 3).reversed_axes();
    let a: Array2<f64> = random_square(3).into_triangular(UPLO::Upper).reversed_axes();
    test2d(UPLO::Lower, a, b, 1e-7);
}
