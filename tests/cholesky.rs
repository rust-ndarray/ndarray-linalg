
extern crate ndarray;
#[macro_use]
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

#[test]
fn cholesky() {
    macro_rules! cholesky {
        ($elem:ty, $rtol:expr) => {
            let a: Array2<$elem> = random_hpd(3);
            println!("a = \n{:?}", a);
            let upper = a.cholesky(UPLO::Upper).unwrap().factor;
            assert_close_l2!(&upper.t().mapv(|elem| elem.conj()).dot(&upper), &a, $rtol);
            let lower = a.cholesky(UPLO::Lower).unwrap().factor;
            assert_close_l2!(&lower.dot(&lower.t().mapv(|elem| elem.conj())), &a, $rtol);
        }
    }
    cholesky!(f64, 1e-9);
    cholesky!(f32, 1e-5);
    cholesky!(c64, 1e-9);
    cholesky!(c32, 1e-5);
}

#[test]
fn cholesky_into_lower_upper() {
    macro_rules! cholesky_into_lower_upper {
        ($elem:ty, $rtol:expr) => {
            let a: Array2<$elem> = random_hpd(3);
            println!("a = \n{:?}", a);
            let upper = a.cholesky(UPLO::Upper).unwrap();
            let lower = a.cholesky(UPLO::Lower).unwrap();
            assert_close_l2!(&upper.factor, &lower.into_upper(), $rtol);
            let upper = a.cholesky(UPLO::Upper).unwrap();
            let lower = a.cholesky(UPLO::Lower).unwrap();
            assert_close_l2!(&lower.factor, &upper.into_lower(), $rtol);
        }
    }
    cholesky_into_lower_upper!(f64, 1e-9);
    cholesky_into_lower_upper!(f32, 1e-5);
    cholesky_into_lower_upper!(c64, 1e-9);
    cholesky_into_lower_upper!(c32, 1e-5);
}

#[test]
fn cholesky_into_inverse() {
    macro_rules! cholesky_into_inverse {
        ($elem:ty, $rtol:expr) => {
            let a: Array2<$elem> = random_hpd(3);
            println!("a = \n{:?}", a);

            let mut inv_upper = a.cholesky(UPLO::Upper).unwrap().into_inverse().unwrap();
            // Fill lower triangular portion with conjugate transpose of upper.
            for row in 0..inv_upper.shape()[0] {
                for col in 0..row {
                    inv_upper[(row, col)] = inv_upper[(col, row)].conj();
                }
            }
            assert_close_l2!(&a.dot(&inv_upper), &Array2::eye(3), $rtol);

            let mut inv_lower = a.cholesky(UPLO::Lower).unwrap().into_inverse().unwrap();
            // Fill upper triangular portion with conjugate transpose of lower.
            for row in 0..(inv_lower.shape()[0] - 1) {
                for col in (row + 1)..inv_lower.shape()[1] {
                    inv_lower[(row, col)] = inv_lower[(col, row)].conj();
                }
            }
            assert_close_l2!(&a.dot(&inv_lower), &Array2::eye(3), $rtol);
        }
    }
    cholesky_into_inverse!(f64, 1e-9);
    cholesky_into_inverse!(f32, 1e-4);
    cholesky_into_inverse!(c64, 1e-9);
    cholesky_into_inverse!(c32, 1e-4);
}

#[test]
fn cholesky_det() {
    macro_rules! cholesky_det {
        ($elem:ty, $rtol:expr) => {
            let a: Array2<$elem> = random_hpd(3);
            println!("a = \n{:?}", a);
            let ln_det = a.eigvalsh(UPLO::Upper).unwrap().mapv(|elem| elem.ln()).scalar_sum();
            let det = ln_det.exp();
            assert_rclose!(a.cholesky(UPLO::Upper).unwrap().ln_det(), ln_det, $rtol);
            assert_rclose!(a.cholesky(UPLO::Lower).unwrap().ln_det(), ln_det, $rtol);
            assert_rclose!(a.cholesky(UPLO::Upper).unwrap().det(), det, $rtol);
            assert_rclose!(a.cholesky(UPLO::Lower).unwrap().det(), det, $rtol);
        }
    }
    cholesky_det!(f64, 1e-9);
    cholesky_det!(f32, 1e-5);
    cholesky_det!(c64, 1e-9);
    cholesky_det!(c32, 1e-5);
}

#[test]
fn cholesky_solve() {
    macro_rules! cholesky_det {
        ($elem:ty, $rtol:expr) => {
            let a: Array2<$elem> = random_hpd(3);
            let x: Array1<$elem> = random(3);
            let b = a.dot(&x);
            println!("a = \n{:?}", a);
            println!("x = \n{:?}", x);
            assert_close_l2!(&a.cholesky(UPLO::Upper).unwrap().solve(&b).unwrap(), &x, $rtol);
            assert_close_l2!(&a.cholesky(UPLO::Lower).unwrap().solve(&b).unwrap(), &x, $rtol);
            assert_close_l2!(&a.cholesky(UPLO::Upper).unwrap().solve_into(b.clone()).unwrap(), &x, $rtol);
            assert_close_l2!(&a.cholesky(UPLO::Lower).unwrap().solve_into(b.clone()).unwrap(), &x, $rtol);
            assert_close_l2!(&a.cholesky(UPLO::Upper).unwrap().solve_mut(&mut b.clone()).unwrap(), &x, $rtol);
            assert_close_l2!(&a.cholesky(UPLO::Lower).unwrap().solve_mut(&mut b.clone()).unwrap(), &x, $rtol);
        }
    }
    cholesky_det!(f64, 1e-9);
    cholesky_det!(f32, 1e-3);
    cholesky_det!(c64, 1e-9);
    cholesky_det!(c32, 1e-3);
}
