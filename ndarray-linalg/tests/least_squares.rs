use approx::AbsDiffEq;
use ndarray::*;
use ndarray_linalg::*;
use num_complex::Complex;

fn c(re: f64, im: f64) -> Complex<f64> {
    Complex::new(re, im)
}

//
// Test cases taken from the scipy test suite for the scipy lstsq function
// https://github.com/scipy/scipy/blob/v1.4.1/scipy/linalg/tests/basic.py
//
#[test]
fn least_squares_exact() {
    let a = array![[1., 20.], [-30., 4.]];
    let bs = vec![
        array![[1., 0.], [0., 1.]],
        array![[1.], [0.]],
        array![[2., 1.], [-30., 4.]],
    ];
    for b in &bs {
        let res = a.least_squares(b).unwrap();
        assert_eq!(res.rank, 2);
        let b_hat = a.dot(&res.solution);
        let rssq = (b - &b_hat).mapv(|x| x.powi(2)).sum_axis(Axis(0));
        assert!(res
            .residual_sum_of_squares
            .unwrap()
            .abs_diff_eq(&rssq, 1e-12));
        assert!(b_hat.abs_diff_eq(&b, 1e-12));
    }
}

#[test]
fn least_squares_overdetermined() {
    let a: Array2<f64> = array![[1., 2.], [4., 5.], [3., 4.]];
    let b: Array1<f64> = array![1., 2., 3.];
    let res = a.least_squares(&b).unwrap();
    assert_eq!(res.rank, 2);
    let b_hat = a.dot(&res.solution);
    let rssq = (&b - &b_hat).mapv(|x| x.powi(2)).sum();
    assert!(res.residual_sum_of_squares.unwrap()[()].abs_diff_eq(&rssq, 1e-12));
    assert!(res
        .solution
        .abs_diff_eq(&array![-0.428571428571429, 0.85714285714285], 1e-12));
}

#[test]
fn least_squares_overdetermined_complex() {
    let a: Array2<c64> = array![
        [c(1., 2.), c(2., 0.)],
        [c(4., 0.), c(5., 0.)],
        [c(3., 0.), c(4., 0.)]
    ];
    let b: Array1<c64> = array![c(1., 0.), c(2., 4.), c(3., 0.)];
    let res = a.least_squares(&b).unwrap();
    assert_eq!(res.rank, 2);
    let b_hat = a.dot(&res.solution);
    let rssq = (&b_hat - &b).mapv(|x| x.powi(2).abs()).sum();
    assert!(res.residual_sum_of_squares.unwrap()[()].abs_diff_eq(&rssq, 1e-12));
    assert!(res.solution.abs_diff_eq(
        &array![
            c(-0.4831460674157303, 0.258426966292135),
            c(0.921348314606741, 0.292134831460674)
        ],
        1e-12
    ));
}

#[test]
fn least_squares_underdetermined() {
    let a: Array2<f64> = array![[1., 2., 3.], [4., 5., 6.]];
    let b: Array1<f64> = array![1., 2.];
    let res = a.least_squares(&b).unwrap();
    assert_eq!(res.rank, 2);
    assert!(res.residual_sum_of_squares.is_none());
    let expected = array![-0.055555555555555, 0.111111111111111, 0.277777777777777];
    assert!(res.solution.abs_diff_eq(&expected, 1e-12));
}

/// This test case tests the underdetermined case for multiple right hand
/// sides. Adapted from scipy lstsq tests.
#[test]
fn least_squares_underdetermined_nrhs() {
    let a: Array2<f64> = array![[1., 2., 3.], [4., 5., 6.]];
    let b: Array2<f64> = array![[1., 1.], [2., 2.]];
    let res = a.least_squares(&b).unwrap();
    assert_eq!(res.rank, 2);
    assert!(res.residual_sum_of_squares.is_none());
    let expected = array![
        [-0.055555555555555, -0.055555555555555],
        [0.111111111111111, 0.111111111111111],
        [0.277777777777777, 0.277777777777777]
    ];
    assert!(res.solution.abs_diff_eq(&expected, 1e-12));
}

//
// Test cases taken from the netlib documentation at
// https://www.netlib.org/lapack/lapacke.html#_calling_code_dgels_code
//
#[test]
fn netlib_lapack_example_for_dgels_1() {
    let a: Array2<f64> = array![
        [1., 1., 1.],
        [2., 3., 4.],
        [3., 5., 2.],
        [4., 2., 5.],
        [5., 4., 3.]
    ];
    let b: Array1<f64> = array![-10., 12., 14., 16., 18.];
    let expected: Array1<f64> = array![2., 1., 1.];
    let result = a.least_squares(&b).unwrap();
    assert!(result.solution.abs_diff_eq(&expected, 1e-12));

    let residual = b - a.dot(&result.solution);
    let resid_ssq = result.residual_sum_of_squares.unwrap();
    assert!((resid_ssq[()] - residual.dot(&residual)).abs() < 1e-12);
}

#[test]
fn netlib_lapack_example_for_dgels_2() {
    let a: Array2<f64> = array![
        [1., 1., 1.],
        [2., 3., 4.],
        [3., 5., 2.],
        [4., 2., 5.],
        [5., 4., 3.]
    ];
    let b: Array1<f64> = array![-3., 14., 12., 16., 16.];
    let expected: Array1<f64> = array![1., 1., 2.];
    let result = a.least_squares(&b).unwrap();
    assert!(result.solution.abs_diff_eq(&expected, 1e-12));

    let residual = b - a.dot(&result.solution);
    let resid_ssq = result.residual_sum_of_squares.unwrap();
    assert!((resid_ssq[()] - residual.dot(&residual)).abs() < 1e-12);
}

#[test]
fn netlib_lapack_example_for_dgels_nrhs() {
    let a: Array2<f64> = array![
        [1., 1., 1.],
        [2., 3., 4.],
        [3., 5., 2.],
        [4., 2., 5.],
        [5., 4., 3.]
    ];
    let b: Array2<f64> = array![[-10., -3.], [12., 14.], [14., 12.], [16., 16.], [18., 16.]];
    let expected: Array2<f64> = array![[2., 1.], [1., 1.], [1., 2.]];
    let result = a.least_squares(&b).unwrap();
    assert!(result.solution.abs_diff_eq(&expected, 1e-12));

    let residual = &b - &a.dot(&result.solution);
    let residual_ssq = residual.mapv(|x| x.powi(2)).sum_axis(Axis(0));
    assert!(result
        .residual_sum_of_squares
        .unwrap()
        .abs_diff_eq(&residual_ssq, 1e-12));
}
