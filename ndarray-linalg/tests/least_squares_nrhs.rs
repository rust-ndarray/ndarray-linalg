/// Solve least square problem `|b - Ax|` with multi-column `b`
use approx::AbsDiffEq;
use ndarray::*;
use ndarray_linalg::*;

/// A is square. `x = A^{-1} b`, `|b - Ax| = 0`
fn test_exact<T: Scalar + Lapack>(a: Array2<T>, b: Array2<T>) {
    assert_eq!(a.layout().unwrap().size(), (3, 3));
    assert_eq!(b.layout().unwrap().size(), (3, 2));

    let result = a.least_squares(&b).unwrap();
    dbg!(&result);
    // unpack result
    let x: Array2<T> = result.solution;
    let residual_l2_square: Array1<T::Real> = result.residual_sum_of_squares.unwrap();

    // must be full-rank
    assert_eq!(result.rank, 3);

    // |b - Ax| == 0
    for &residual in &residual_l2_square {
        assert!(residual < T::real(1.0e-4));
    }

    // b == Ax
    let ax = a.dot(&x);
    assert_close_l2!(&b, &ax, T::real(1.0e-4));
}

macro_rules! impl_exact {
    ($scalar:ty) => {
        paste::item! {
            #[test]
            fn [<least_squares_ $scalar _exact_ac_bc>]() {
                let a: Array2<$scalar> = random((3, 3));
                let b: Array2<$scalar> = random((3, 2));
                test_exact(a, b)
            }

            /* Unsupported currently. See https://github.com/rust-ndarray/ndarray-linalg/issues/234

            #[test]
            fn [<least_squares_ $scalar _exact_ac_bf>]() {
                let a: Array2<$scalar> = random((3, 3));
                let b: Array2<$scalar> = random((3, 2).f());
                test_exact(a, b)
            }

            #[test]
            fn [<least_squares_ $scalar _exact_af_bc>]() {
                let a: Array2<$scalar> = random((3, 3).f());
                let b: Array2<$scalar> = random((3, 2));
                test_exact(a, b)
            }

            */

            #[test]
            fn [<least_squares_ $scalar _exact_af_bf>]() {
                let a: Array2<$scalar> = random((3, 3).f());
                let b: Array2<$scalar> = random((3, 2).f());
                test_exact(a, b)
            }
        }
    };
}

impl_exact!(f32);
impl_exact!(f64);
impl_exact!(c32);
impl_exact!(c64);

/// #column < #row case.
/// Linear problem is overdetermined, `|b - Ax| > 0`.
fn test_overdetermined<T: Scalar + Lapack>(a: Array2<T>, bs: Array2<T>)
where
    T::Real: AbsDiffEq<Epsilon = T::Real>,
{
    assert_eq!(a.layout().unwrap().size(), (4, 3));
    assert_eq!(bs.layout().unwrap().size(), (4, 2));

    let result = a.least_squares(&bs).unwrap();
    // unpack result
    let xs = result.solution;
    let residual_l2_square = result.residual_sum_of_squares.unwrap();

    // Must be full-rank
    assert_eq!(result.rank, 3);

    for j in 0..2 {
        let b = bs.index_axis(Axis(1), j);
        let x = xs.index_axis(Axis(1), j);
        let residual = &b - &a.dot(&x);
        let residual_l2_sq = residual_l2_square[j];
        assert!(residual_l2_sq.abs_diff_eq(&residual.norm_l2().powi(2), T::real(1.0e-4)));

        // `|residual| < |b|`
        assert!(residual.norm_l2() < b.norm_l2());
    }
}

macro_rules! impl_overdetermined {
    ($scalar:ty) => {
        paste::item! {
            #[test]
            fn [<least_squares_ $scalar _overdetermined_ac_bc>]() {
                let a: Array2<$scalar> = random((4, 3));
                let b: Array2<$scalar> = random((4, 2));
                test_overdetermined(a, b)
            }

            /* Unsupported currently. See https://github.com/rust-ndarray/ndarray-linalg/issues/234

            #[test]
            fn [<least_squares_ $scalar _overdetermined_af_bc>]() {
                let a: Array2<$scalar> = random((4, 3).f());
                let b: Array2<$scalar> = random((4, 2));
                test_overdetermined(a, b)
            }

            #[test]
            fn [<least_squares_ $scalar _overdetermined_ac_bf>]() {
                let a: Array2<$scalar> = random((4, 3));
                let b: Array2<$scalar> = random((4, 2).f());
                test_overdetermined(a, b)
            }

            */

            #[test]
            fn [<least_squares_ $scalar _overdetermined_af_bf>]() {
                let a: Array2<$scalar> = random((4, 3).f());
                let b: Array2<$scalar> = random((4, 2).f());
                test_overdetermined(a, b)
            }
        }
    };
}

impl_overdetermined!(f32);
impl_overdetermined!(f64);
impl_overdetermined!(c32);
impl_overdetermined!(c64);

/// #column > #row case.
/// Linear problem is underdetermined, `|b - Ax| = 0` and `x` is not unique
fn test_underdetermined<T: Scalar + Lapack>(a: Array2<T>, b: Array2<T>) {
    assert_eq!(a.layout().unwrap().size(), (3, 4));
    assert_eq!(b.layout().unwrap().size(), (3, 2));

    let result = a.least_squares(&b).unwrap();
    assert_eq!(result.rank, 3);
    assert!(result.residual_sum_of_squares.is_none());

    // b == Ax
    let x = result.solution;
    let ax = a.dot(&x);
    assert_close_l2!(&b, &ax, T::real(1.0e-4));
}

macro_rules! impl_underdetermined {
    ($scalar:ty) => {
        paste::item! {
            #[test]
            fn [<least_squares_ $scalar _underdetermined_ac_bc>]() {
                let a: Array2<$scalar> = random((3, 4));
                let b: Array2<$scalar> = random((3, 2));
                test_underdetermined(a, b)
            }

            /* Unsupported currently. See https://github.com/rust-ndarray/ndarray-linalg/issues/234

            #[test]
            fn [<least_squares_ $scalar _underdetermined_af_bc>]() {
                let a: Array2<$scalar> = random((3, 4).f());
                let b: Array2<$scalar> = random((3, 2));
                test_underdetermined(a, b)
            }

            #[test]
            fn [<least_squares_ $scalar _underdetermined_ac_bf>]() {
                let a: Array2<$scalar> = random((3, 4));
                let b: Array2<$scalar> = random((3, 2).f());
                test_underdetermined(a, b)
            }

            */

            #[test]
            fn [<least_squares_ $scalar _underdetermined_af_bf>]() {
                let a: Array2<$scalar> = random((3, 4).f());
                let b: Array2<$scalar> = random((3, 2).f());
                test_underdetermined(a, b)
            }
        }
    };
}

impl_underdetermined!(f32);
impl_underdetermined!(f64);
impl_underdetermined!(c32);
impl_underdetermined!(c64);
