/// Solve least square problem `|b - Ax|`
use approx::AbsDiffEq;
use ndarray::*;
use ndarray_linalg::*;

/// A is square. `x = A^{-1} b`, `|b - Ax| = 0`
fn test_exact<T: Scalar + Lapack>(a: Array2<T>) {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let b: Array1<T> = random_using(3, &mut rng);
    let result = a.least_squares(&b).unwrap();
    // unpack result
    let x = result.solution;
    let residual_l2_square = result.residual_sum_of_squares.unwrap()[()];

    // must be full-rank
    assert_eq!(result.rank, 3);

    // |b - Ax| == 0
    assert!(residual_l2_square < T::real(1.0e-4));

    // b == Ax
    let ax = a.dot(&x);
    assert_close_max!(&b, &ax, T::real(1.0e-4));
}

macro_rules! impl_exact {
    ($scalar:ty) => {
        paste::item! {
            #[test]
            fn [<least_squares_ $scalar _exact>]() {
                let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
                let a: Array2<$scalar> = random_using((3, 3), &mut rng);
                test_exact(a)
            }

            #[test]
            fn [<least_squares_ $scalar _exact_t>]() {
                let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
                let a: Array2<$scalar> = random_using((3, 3).f(), &mut rng);
                test_exact(a)
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
fn test_overdetermined<T: Scalar + Lapack>(a: Array2<T>)
where
    T::Real: AbsDiffEq<Epsilon = T::Real>,
{
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let b: Array1<T> = random_using(4, &mut rng);
    let result = a.least_squares(&b).unwrap();
    // unpack result
    let x = result.solution;
    let residual_l2_square = result.residual_sum_of_squares.unwrap()[()];

    // Must be full-rank
    assert_eq!(result.rank, 3);

    // eval `residual = b - Ax`
    let residual = &b - &a.dot(&x);
    assert!(residual_l2_square.abs_diff_eq(&residual.norm_l2().powi(2), T::real(1.0e-4)));

    // `|residual| < |b|`
    assert!(residual.norm_l2() < b.norm_l2());
}

macro_rules! impl_overdetermined {
    ($scalar:ty) => {
        paste::item! {
            #[test]
            fn [<least_squares_ $scalar _overdetermined>]() {
                let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
                let a: Array2<$scalar> = random_using((4, 3), &mut rng);
                test_overdetermined(a)
            }

            #[test]
            fn [<least_squares_ $scalar _overdetermined_t>]() {
                let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
                let a: Array2<$scalar> = random_using((4, 3).f(), &mut rng);
                test_overdetermined(a)
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
fn test_underdetermined<T: Scalar + Lapack>(a: Array2<T>) {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let b: Array1<T> = random_using(3, &mut rng);
    let result = a.least_squares(&b).unwrap();
    assert_eq!(result.rank, 3);
    assert!(result.residual_sum_of_squares.is_none());

    // b == Ax
    let x = result.solution;
    let ax = a.dot(&x);
    assert_close_max!(&b, &ax, T::real(1.0e-4));
}

macro_rules! impl_underdetermined {
    ($scalar:ty) => {
        paste::item! {
            #[test]
            fn [<least_squares_ $scalar _underdetermined>]() {
                let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
                let a: Array2<$scalar> = random_using((3, 4), &mut rng);
                test_underdetermined(a)
            }

            #[test]
            fn [<least_squares_ $scalar _underdetermined_t>]() {
                let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
                let a: Array2<$scalar> = random_using((3, 4).f(), &mut rng);
                test_underdetermined(a)
            }
        }
    };
}

impl_underdetermined!(f32);
impl_underdetermined!(f64);
impl_underdetermined!(c32);
impl_underdetermined!(c64);
