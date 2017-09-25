extern crate ndarray;
#[macro_use]
extern crate ndarray_linalg;
extern crate num_traits;

use ndarray::*;
use ndarray_linalg::*;
use num_traits::Zero;

fn det_3x3<A, S>(a: ArrayBase<S, Ix2>) -> A
where
    A: Scalar,
    S: Data<Elem = A>,
{
    a[(0, 0)] * a[(1, 1)] * a[(2, 2)] + a[(0, 1)] * a[(1, 2)] * a[(2, 0)] + a[(0, 2)] * a[(1, 0)] * a[(2, 1)] -
        a[(0, 2)] * a[(1, 1)] * a[(2, 0)] - a[(0, 1)] * a[(1, 0)] * a[(2, 2)] - a[(0, 0)] * a[(1, 2)] * a[(2, 1)]
}

#[test]
fn det_zero() {
    macro_rules! det_zero {
        ($elem:ty) => {
            let a: Array2<$elem> = array![[Zero::zero()]];
            assert_eq!(a.det().unwrap(), Zero::zero());
            assert_eq!(a.det_into().unwrap(), Zero::zero());
        }
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
            assert!(a.det_into().is_err());
        }
    }
    for &shape in &[(1, 2).into_shape(), (1, 2).f()] {
        det_zero_nonsquare!(f64, shape);
        det_zero_nonsquare!(f32, shape);
        det_zero_nonsquare!(c64, shape);
        det_zero_nonsquare!(c32, shape);
    }
}

#[test]
fn det() {
    macro_rules! det {
        ($elem:ty, $shape:expr, $rtol:expr) => {
            let a: Array2<$elem> = random($shape);
            println!("a = \n{:?}", a);
            let det = det_3x3(a.view());
            assert_rclose!(a.factorize().unwrap().det().unwrap(), det, $rtol);
            assert_rclose!(a.factorize().unwrap().det_into().unwrap(), det, $rtol);
            assert_rclose!(a.det().unwrap(), det, $rtol);
            assert_rclose!(a.det_into().unwrap(), det, $rtol);
        }
    }
    for &shape in &[(3, 3).into_shape(), (3, 3).f()] {
        det!(f64, shape, 1e-9);
        det!(f32, shape, 1e-4);
        det!(c64, shape, 1e-9);
        det!(c32, shape, 1e-4);
    }
}

#[test]
fn det_nonsquare() {
    macro_rules! det_nonsquare {
        ($elem:ty, $shape:expr) => {
            let a: Array2<$elem> = random($shape);
            assert!(a.factorize().unwrap().det().is_err());
            assert!(a.factorize().unwrap().det_into().is_err());
            assert!(a.det().is_err());
            assert!(a.det_into().is_err());
        }
    }
    for &shape in &[(1, 2).into_shape(), (1, 2).f(), (2, 1).into_shape(), (2, 1).f()] {
        det_nonsquare!(f64, shape);
        det_nonsquare!(f32, shape);
        det_nonsquare!(c64, shape);
        det_nonsquare!(c32, shape);
    }
}
