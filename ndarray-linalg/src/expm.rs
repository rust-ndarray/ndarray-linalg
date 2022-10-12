use crate::types;
use condest::Normest1;
use ndarray::prelude::*;
use statrs::function::factorial::{binomial, factorial};

// These constants are hard-coded from Al-Mohy & Higham
const THETA_3: f64 = 1.495585217958292e-2;
const THETA_5: f64 = 2.539398330063230e-1;
const THETA_7: f64 = 9.504178996162932e-1;
const THETA_9: f64 = 2.097847961257068e0;
const THETA_13: f64 = 4.25; // Alg 5.1

// Corresponds to even powers of x. Note that the denominator coefficients are same magnitude but opposite sign. Zeroth order coefficient is 1.
const PADE_COEFFS: [f64; 7] = [
    0.5,
    11. / 600.,
    3. / 18400.,
    1. / 1932000.,
    1. / 1585785600.,
    1. / 3953892096000.,
    1. / 64764752532480000.,
];

// helper function used in Al-M & H. Name is unchanged for future reference.
fn ell(A: Array2<f64>, m: u64, normestimator: Normest1) -> i32 {
    1
}

/// Calculates the leading term of the error series for the [m/m] Pade approximation to exp(x).
fn pade_error_coefficient(m: u64) -> f64 {
    1.0 / (binomial(2 * m, m) * factorial(2 * m + 1))
}

fn paterson_stockmeyer() -> () {}

/// This function is based on the scale-and-squaring algorithm by
pub fn expm(A: Array2<i32>) -> Array2<i32> {
    arr2(&[[2]])
}

mod tests {
    use ndarray::{linalg::Dot, *};

    /// Compares expm acting on a matrix with random eigenvalues (drawn from
    /// Gaussians) and with random eigenvectors (drawn from Haar distribution)
    /// to the exact answer. The exact answer is done by exponentiating each
    /// diagonal entry in the eigenvalue matrix before conjugating with the
    /// eigenvector matrices. In other words, let A = U D U^\dagger, then
    /// because e^A = e^(U D U^\dagger) = U (e^D) U^dagger. We use expm
    /// to compute e^A and normal floating point exponentiation to compute e^D
    #[test]
    fn expm_test_gaussian_random_input() {
        let a: Array2<i32> = arr2(&[[1, 2], [3, 4]]);
        let b: Array2<i32> = arr2(&[[0, 1], [1, 0]]);
        println!("matrix multiplication? {:?}", a.dot(&b));
        // let eigenvals = diagonal::Diagonal::from([1,2,3,4,5]);
        // let eigenvecs = haar_random(dim);
        // let exact = floating_exp(D);
        // let diff = expm(U D U.conj().T) - U F U.conj().T;
    }
}
