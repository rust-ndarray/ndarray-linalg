use ndarray::*;
use ndarray_linalg::*;
use num_traits::{Float, One, Zero};

#[test]
fn deth_empty() {
    macro_rules! deth_empty {
        ($elem:ty) => {
            let a: Array2<$elem> = Array2::zeros((0, 0));
            assert_eq!(a.factorizeh().unwrap().deth(), One::one());
            assert_eq!(a.factorizeh().unwrap().sln_deth(), (One::one(), Zero::zero()));
            assert_eq!(a.factorizeh().unwrap().deth_into(), One::one());
            assert_eq!(
                a.factorizeh().unwrap().sln_deth_into(),
                (One::one(), Zero::zero())
            );
            assert_eq!(a.deth().unwrap(), One::one());
            assert_eq!(a.sln_deth().unwrap(), (One::one(), Zero::zero()));
            assert_eq!(a.clone().deth_into().unwrap(), One::one());
            assert_eq!(a.sln_deth_into().unwrap(), (One::one(), Zero::zero()));
        };
    }
    deth_empty!(f64);
    deth_empty!(f32);
    deth_empty!(c64);
    deth_empty!(c32);
}

#[test]
fn deth_zero() {
    macro_rules! deth_zero {
        ($elem:ty) => {
            let a: Array2<$elem> = Array2::zeros((1, 1));
            assert_eq!(a.deth().unwrap(), Zero::zero());
            assert_eq!(a.sln_deth().unwrap(), (Zero::zero(), Float::neg_infinity()));
            assert_eq!(a.clone().deth_into().unwrap(), Zero::zero());
            assert_eq!(a.sln_deth_into().unwrap(), (Zero::zero(), Float::neg_infinity()));
        };
    }
    deth_zero!(f64);
    deth_zero!(f32);
    deth_zero!(c64);
    deth_zero!(c32);
}

#[test]
fn deth_zero_nonsquare() {
    macro_rules! deth_zero_nonsquare {
        ($elem:ty, $shape:expr) => {
            let a: Array2<$elem> = Array2::zeros($shape);
            assert!(a.deth().is_err());
            assert!(a.sln_deth().is_err());
            assert!(a.clone().deth_into().is_err());
            assert!(a.sln_deth_into().is_err());
        };
    }
    for &shape in &[(1, 2).into_shape(), (1, 2).f()] {
        deth_zero_nonsquare!(f64, shape);
        deth_zero_nonsquare!(f32, shape);
        deth_zero_nonsquare!(c64, shape);
        deth_zero_nonsquare!(c32, shape);
    }
}

#[test]
fn deth() {
    macro_rules! deth {
        ($elem:ty, $rows:expr, $atol:expr) => {
            let a: Array2<$elem> = random_hermite($rows);
            println!("a = \n{:?}", a);

            // Compute determinant from eigenvalues.
            let (sign, ln_det) = a.eigvalsh(UPLO::Upper).unwrap().iter().fold(
                (<$elem as Scalar>::Real::one(), <$elem as Scalar>::Real::zero()),
                |(sign, ln_det), eigval| (sign * eigval.signum(), ln_det + eigval.abs().ln()),
            );
            let det = sign * ln_det.exp();
            assert_aclose!(det, a.eigvalsh(UPLO::Upper).unwrap().iter().product(), $atol);

            assert_aclose!(a.factorizeh().unwrap().deth(), det, $atol);
            {
                let result = a.factorizeh().unwrap().sln_deth();
                assert_aclose!(result.0, sign, $atol);
                assert_aclose!(result.1, ln_det, $atol);
            }
            assert_aclose!(a.factorizeh().unwrap().deth_into(), det, $atol);
            {
                let result = a.factorizeh().unwrap().sln_deth_into();
                assert_aclose!(result.0, sign, $atol);
                assert_aclose!(result.1, ln_det, $atol);
            }
            assert_aclose!(a.deth().unwrap(), det, $atol);
            {
                let result = a.sln_deth().unwrap();
                assert_aclose!(result.0, sign, $atol);
                assert_aclose!(result.1, ln_det, $atol);
            }
            assert_aclose!(a.clone().deth_into().unwrap(), det, $atol);
            {
                let result = a.sln_deth_into().unwrap();
                assert_aclose!(result.0, sign, $atol);
                assert_aclose!(result.1, ln_det, $atol);
            }
        };
    }
    for rows in 1..6 {
        deth!(f64, rows, 1e-9);
        deth!(f32, rows, 1e-3);
        deth!(c64, rows, 1e-9);
        deth!(c32, rows, 1e-3);
    }
}

#[test]
fn deth_nonsquare() {
    macro_rules! deth_nonsquare {
        ($elem:ty, $shape:expr) => {
            let a: Array2<$elem> = Array2::zeros($shape);
            assert!(a.factorizeh().is_err());
            assert!(a.deth().is_err());
            assert!(a.sln_deth().is_err());
            assert!(a.clone().deth_into().is_err());
            assert!(a.sln_deth_into().is_err());
        };
    }
    for &dims in &[(1, 0), (1, 2), (2, 1), (2, 3)] {
        for &shape in &[dims.into_shape(), dims.f()] {
            deth_nonsquare!(f64, shape);
            deth_nonsquare!(f32, shape);
            deth_nonsquare!(c64, shape);
            deth_nonsquare!(c32, shape);
        }
    }
}
