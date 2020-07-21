/// Solve least square problem `|b - Ax|`
use approx::AbsDiffEq;
use ndarray::*;
use ndarray_linalg::*;

/// A is square. `x = A^{-1} b`, `|b - Ax| = 0`
#[test]
fn least_squares_exact() {
    let a: Array2<f64> = random((3, 3));
    let b: Array1<f64> = random(3);
    let result = a.least_squares(&b).unwrap();
    // unpack result
    let x = result.solution;
    let residual_l2_square = result.residual_sum_of_squares.unwrap()[()];

    // must be full-rank
    assert_eq!(result.rank, 3);

    // |b - Ax| == 0
    assert!(residual_l2_square < 1.0e-7);

    // b == Ax
    let ax = a.dot(&x);
    assert_close_l2!(&b, &ax, 1.0e-7);
}

/// #column < #row case.
/// Linear problem is overdetermined, `|b - Ax| > 0`.
#[test]
fn least_squares_overdetermined() {
    let a: Array2<f64> = random((4, 3));
    let b: Array1<f64> = random(4);
    let result = a.least_squares(&b).unwrap();
    // unpack result
    let x = result.solution;
    let residual_l2_square = result.residual_sum_of_squares.unwrap()[()];

    // Must be full-rank
    assert_eq!(result.rank, 3);

    // eval `residual = b - Ax`
    let residual = &b - &a.dot(&x);
    assert!(residual_l2_square.abs_diff_eq(&residual.norm_l2().powi(2), 1e-12));

    // `|residual| < |b|`
    assert!(residual.norm_l2() < b.norm_l2());
}

/// #column > #row case.
/// Linear problem is underdetermined, `|b - Ax| = 0` and `x` is not unique
#[test]
fn least_squares_underdetermined() {
    let a: Array2<f64> = random((3, 4));
    let b: Array1<f64> = random(3);
    let result = a.least_squares(&b).unwrap();
    assert_eq!(result.rank, 3);
    assert!(result.residual_sum_of_squares.is_none());

    // b == Ax
    let x = result.solution;
    let ax = a.dot(&x);
    assert_close_l2!(&b, &ax, 1.0e-7);
}
