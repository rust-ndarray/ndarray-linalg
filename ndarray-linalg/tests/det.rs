use ndarray::*;
use ndarray_linalg::*;
use num_traits::{Float, One, Zero};

/// Returns the matrix with the specified `row` and `col` removed.
fn matrix_minor<A, S>(a: &ArrayBase<S, Ix2>, (row, col): (usize, usize)) -> Array2<A>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    let mut select_rows = (0..a.nrows()).collect::<Vec<_>>();
    select_rows.remove(row);
    let mut select_cols = (0..a.ncols()).collect::<Vec<_>>();
    select_cols.remove(col);
    a.select(Axis(0), &select_rows)
        .select(Axis(1), &select_cols)
}

/// Computes the determinant of matrix `a`.
///
/// Note: This implementation is written to be clearly correct so that it's
/// useful for verification, but it's very inefficient.
fn det_naive<A, S>(a: &ArrayBase<S, Ix2>) -> A
where
    A: Scalar,
    S: Data<Elem = A>,
{
    assert_eq!(a.nrows(), a.ncols());
    match a.ncols() {
        0 => A::one(),
        1 => a[(0, 0)],
        cols => (0..cols)
            .map(|col| {
                let sign = if col % 2 == 0 { A::one() } else { -A::one() };
                sign * a[(0, col)] * det_naive(&matrix_minor(a, (0, col)))
            })
            .fold(A::zero(), |sum, subdet| sum + subdet),
    }
}

#[test]
fn det_empty() {
    macro_rules! det_empty {
        ($elem:ty) => {
            let a: Array2<$elem> = Array2::zeros((0, 0));
            let det = One::one();
            let (sign, ln_det) = (One::one(), Zero::zero());
            assert_eq!(a.factorize().unwrap().det().unwrap(), det);
            assert_eq!(a.factorize().unwrap().sln_det().unwrap(), (sign, ln_det));
            assert_eq!(a.factorize().unwrap().det_into().unwrap(), det);
            assert_eq!(
                a.factorize().unwrap().sln_det_into().unwrap(),
                (sign, ln_det)
            );
            assert_eq!(a.det().unwrap(), det);
            assert_eq!(a.sln_det().unwrap(), (sign, ln_det));
            assert_eq!(a.clone().det_into().unwrap(), det);
            assert_eq!(a.sln_det_into().unwrap(), (sign, ln_det));
        };
    }
    det_empty!(f64);
    det_empty!(f32);
    det_empty!(c64);
    det_empty!(c32);
}

#[test]
fn det_zero() {
    macro_rules! det_zero {
        ($elem:ty) => {
            let a: Array2<$elem> = Array2::zeros((1, 1));
            let det = Zero::zero();
            let (sign, ln_det) = (Zero::zero(), Float::neg_infinity());
            assert_eq!(a.det().unwrap(), det);
            assert_eq!(a.sln_det().unwrap(), (sign, ln_det));
            assert_eq!(a.clone().det_into().unwrap(), det);
            assert_eq!(a.sln_det_into().unwrap(), (sign, ln_det));
        };
    }
    det_zero!(f64);
    det_zero!(f32);
    det_zero!(c64);
    det_zero!(c32);
}

#[test]
fn det_zero_nonsquare() {
    macro_rules! det_zero_nonsquare {
        ($elem:ty, $shape:expr) => {
            let a: Array2<$elem> = Array2::zeros($shape);
            assert!(a.det().is_err());
            assert!(a.sln_det().is_err());
            assert!(a.clone().det_into().is_err());
            assert!(a.sln_det_into().is_err());
        };
    }
    for &shape in &[(1, 2).into_shape_with_order(), (1, 2).f()] {
        det_zero_nonsquare!(f64, shape);
        det_zero_nonsquare!(f32, shape);
        det_zero_nonsquare!(c64, shape);
        det_zero_nonsquare!(c32, shape);
    }
}

#[test]
fn det() {
    fn det_impl<A>(a: Array2<A>, rtol: A::Real)
    where
        A: Scalar + Lapack,
    {
        let det = det_naive(&a);
        let sign = det.div_real(det.abs());
        let ln_det = Float::ln(det.abs());
        assert_rclose!(a.factorize().unwrap().det().unwrap(), det, rtol);
        {
            let result = a.factorize().unwrap().sln_det().unwrap();
            assert_rclose!(result.0, sign, rtol);
            assert_rclose!(result.1, ln_det, rtol);
        }
        assert_rclose!(a.factorize().unwrap().det_into().unwrap(), det, rtol);
        {
            let result = a.factorize().unwrap().sln_det_into().unwrap();
            assert_rclose!(result.0, sign, rtol);
            assert_rclose!(result.1, ln_det, rtol);
        }
        assert_rclose!(a.det().unwrap(), det, rtol);
        {
            let result = a.sln_det().unwrap();
            assert_rclose!(result.0, sign, rtol);
            assert_rclose!(result.1, ln_det, rtol);
        }
        assert_rclose!(a.clone().det_into().unwrap(), det, rtol);
        {
            let result = a.sln_det_into().unwrap();
            assert_rclose!(result.0, sign, rtol);
            assert_rclose!(result.1, ln_det, rtol);
        }
    }
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    for rows in 1..5 {
        det_impl(random_regular_using::<f64, _>(rows, &mut rng), 1e-9);
        det_impl(random_regular_using::<f32, _>(rows, &mut rng), 1e-4);
        det_impl(random_regular_using::<c64, _>(rows, &mut rng), 1e-9);
        det_impl(random_regular_using::<c32, _>(rows, &mut rng), 1e-4);
        det_impl(
            random_regular_using::<f64, _>(rows, &mut rng)
                .t()
                .to_owned(),
            1e-9,
        );
        det_impl(
            random_regular_using::<f32, _>(rows, &mut rng)
                .t()
                .to_owned(),
            1e-4,
        );
        det_impl(
            random_regular_using::<c64, _>(rows, &mut rng)
                .t()
                .to_owned(),
            1e-9,
        );
        det_impl(
            random_regular_using::<c32, _>(rows, &mut rng)
                .t()
                .to_owned(),
            1e-4,
        );
    }
}

#[test]
fn det_nonsquare() {
    macro_rules! det_nonsquare {
        ($elem:ty, $shape:expr) => {
            let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
            let a: Array2<$elem> = random_using($shape, &mut rng);
            assert!(a.factorize().unwrap().det().is_err());
            assert!(a.factorize().unwrap().sln_det().is_err());
            assert!(a.factorize().unwrap().det_into().is_err());
            assert!(a.factorize().unwrap().sln_det_into().is_err());
            assert!(a.det().is_err());
            assert!(a.sln_det().is_err());
            assert!(a.clone().det_into().is_err());
            assert!(a.sln_det_into().is_err());
        };
    }
    for &dims in &[(1, 0), (1, 2), (2, 1), (2, 3)] {
        for &shape in &[dims.into_shape_with_order(), dims.f()] {
            det_nonsquare!(f64, shape);
            det_nonsquare!(f32, shape);
            det_nonsquare!(c64, shape);
            det_nonsquare!(c32, shape);
        }
    }
}
