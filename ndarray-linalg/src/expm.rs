use std::ops::MulAssign;

use crate::{normest1::normest, types, Inverse, OperationNorm};
use cauchy::Scalar;
use lax::{Lapack, NormType};
use ndarray::{linalg::Dot, prelude::*};
use num_complex::{Complex, Complex32 as c32, Complex64 as c64};
use num_traits::{real::Real, Pow};
extern crate statrs;
use statrs::{
    function::factorial::{binomial, factorial},
    statistics::Statistics,
};

// These constants are hard-coded from Al-Mohy & Higham
const THETA_3: f64 = 1.495585217958292e-2;
const THETA_5: f64 = 2.539398330063230e-1;
const THETA_7: f64 = 9.504178996162932e-1;
const THETA_9: f64 = 2.097847961257068e0;
const THETA_13: f64 = 4.25; // Alg 5.1

// The Pade Coefficients for the numerator of the diagonal approximation to Exp[x]. Computed via Mathematica/WolframAlpha.
// Note that the denominator has the same coefficients but odd powers have an opposite sign.
// Coefficients are also stored via the power of x, so for example the numerator would look like
// x^0 * PADE_COEFF_M[0] + x^1 * PADE_COEFF_M[1] + ... + x^m * PADE_COEFF_M[M]
const PADE_COEFFS_3: [f64; 4] = [1., 0.5, 0.1, 1. / 120.];

const PADE_COEFFS_5: [f64; 6] = [1., 0.5, 1. / 9., 1. / 72., 1. / 1_008., 1. / 30_240.];

const PADE_COEFFS_7: [f64; 8] = [
    1.,
    0.5,
    3. / 26.,
    5. / 312.,
    5. / 3_432.,
    1. / 11_440.,
    1. / 308_880.,
    1. / 17_297_280.,
];

const PADE_COEFFS_9: [f64; 10] = [
    1.,
    0.5,
    2. / 17.,
    7. / 408.,
    7. / 4_080.,
    1. / 8_160.,
    1. / 159_120.,
    1. / 4_455_360.,
    1. / 196_035_840.,
    1. / 17_643_225_600.,
];

const PADE_COEFFS_13: [f64; 14] = [
    1.,
    0.5,
    3. / 25.,
    11. / 600.,
    11. / 5_520.,
    3. / 18_400.,
    1. / 96_600.,
    1. / 1_932_000.,
    1. / 48_944_000.,
    1. / 1_585_785_600.,
    1. / 67_395_888_000.,
    1. / 3_953_892_096_000.,
    1. / 355_850_288_640_000.,
    1. / 64_764_752_532_480_000.,
];

// These are the ones used in scipy
// const PADE_COEFFS_13: [f64; 14] = [
//     64764752532480000.,
//     32382376266240000.,
//     7771770303897600.,
//     1187353796428800.,
//     129060195264000.,
//     10559470521600.,
//     670442572800.,
//     33522128640.,
//     1323241920.,
//     40840800.,
//     960960.,
//     16380.,
//     182.,
//     1.,
// ];

fn pade_approximation_3<S: Scalar<Real = f64> + Lapack>(
    a_1: &Array2<S>,
    a_2: &Array2<S>,
) -> Array2<S> {
    let mut evens: Array2<S> = Array2::<S>::eye(a_1.nrows());
    // evens.mapv_inplace(|x| x * S::from_real(PADE_COEFFS_3[0]));
    evens.scaled_add(S::from_real(PADE_COEFFS_3[2]), a_2);

    let mut odds: Array2<S> = Array2::<S>::eye(a_1.nrows());
    odds.mapv_inplace(|x| x * S::from_real(PADE_COEFFS_3[1]));
    odds.scaled_add(S::from_real(PADE_COEFFS_3[3]), a_2);
    odds = odds.dot(a_1);

    odds.mapv_inplace(|x| -x);
    let inverted = (&odds + &evens).inv().unwrap();
    odds.mapv_inplace(|x| -x);
    inverted.dot(&(odds + evens))
}

fn pade_approximation_5<S: Scalar<Real = f64> + Lapack>(
    a_1: &Array2<S>,
    a_2: &Array2<S>,
    a_4: &Array2<S>,
) -> Array2<S> {
    let mut evens: Array2<S> = Array2::<S>::eye(a_1.nrows());
    // evens.mapv_inplace(|x| S::from_real(PADE_COEFFS_5[0]) * x);
    evens.scaled_add(S::from_real(PADE_COEFFS_5[2]), a_2);
    evens.scaled_add(S::from_real(PADE_COEFFS_5[4]), a_4);

    let mut odds: Array2<S> = Array::eye(a_1.nrows());
    odds.mapv_inplace(|x| S::from_real(PADE_COEFFS_5[1]) * x);
    odds.scaled_add(S::from_real(PADE_COEFFS_5[3]), a_2);
    odds.scaled_add(S::from_real(PADE_COEFFS_5[5]), a_4);
    odds = odds.dot(a_1);

    odds.mapv_inplace(|x| -x);
    let inverted = (&odds + &evens).inv().unwrap();
    odds.mapv_inplace(|x| -x);
    inverted.dot(&(odds + evens))
}

fn pade_approximation_7<S: Scalar<Real = f64> + Lapack>(
    a_1: &Array2<S>,
    a_2: &Array2<S>,
    a_4: &Array2<S>,
    a_6: &Array2<S>,
) -> Array2<S> {
    let mut evens: Array2<S> = Array::eye(a_1.nrows());
    // evens.mapv_inplace(|x| S::from_real(PADE_COEFFS_7[0]) * x);
    evens.scaled_add(S::from_real(PADE_COEFFS_7[2]), a_2);
    evens.scaled_add(S::from_real(PADE_COEFFS_7[4]), a_4);
    evens.scaled_add(S::from_real(PADE_COEFFS_7[6]), a_6);

    let mut odds: Array2<S> = Array::eye(a_1.nrows());
    odds.mapv_inplace(|x| S::from_real(PADE_COEFFS_7[1]) * x);
    odds.scaled_add(S::from_real(PADE_COEFFS_7[3]), a_2);
    odds.scaled_add(S::from_real(PADE_COEFFS_7[5]), a_4);
    odds.scaled_add(S::from_real(PADE_COEFFS_7[7]), a_6);
    odds = odds.dot(a_1);

    odds.mapv_inplace(|x| -x);
    let inverted = (&odds + &evens).inv().unwrap();
    odds.mapv_inplace(|x| -x);
    inverted.dot(&(odds + evens))
}

fn pade_approximation_9<S: Scalar<Real = f64> + Lapack>(
    a_1: &Array2<S>,
    a_2: &Array2<S>,
    a_4: &Array2<S>,
    a_6: &Array2<S>,
    a_8: &Array2<S>,
) -> Array2<S> {
    let mut evens: Array2<S> = Array::eye(a_1.nrows());
    // evens.mapv_inplace(|x| S::from_real(PADE_COEFFS_9[0]) * x);
    evens.scaled_add(S::from_real(PADE_COEFFS_9[2]), a_2);
    evens.scaled_add(S::from_real(PADE_COEFFS_9[4]), a_4);
    evens.scaled_add(S::from_real(PADE_COEFFS_9[6]), a_6);
    evens.scaled_add(S::from_real(PADE_COEFFS_9[8]), a_8);

    let mut odds: Array2<S> = Array::eye(a_1.nrows());
    odds.mapv_inplace(|x| S::from_real(PADE_COEFFS_9[1]) * x);
    odds.scaled_add(S::from_real(PADE_COEFFS_9[3]), a_2);
    odds.scaled_add(S::from_real(PADE_COEFFS_9[5]), a_4);
    odds.scaled_add(S::from_real(PADE_COEFFS_9[7]), a_6);
    odds.scaled_add(S::from_real(PADE_COEFFS_9[9]), a_8);
    odds = odds.dot(a_1);

    odds.mapv_inplace(|x| -x);
    let inverted: Array2<S> = (&odds + &evens).inv().unwrap();
    odds.mapv_inplace(|x| -x);
    inverted.dot(&(odds + evens))
}

// Note: all input matrices should be scaled in the main expm
// function. 
fn pade_approximation_13<S: Scalar<Real = f64> + Lapack>(
    a_1: &Array2<S>,
    a_2: &Array2<S>,
    a_4: &Array2<S>,
    a_6: &Array2<S>,
) -> Array2<S> {
    let mut evens_1: Array2<S> = Array::eye(a_1.nrows());
    evens_1.mapv_inplace(|x| S::from_real(PADE_COEFFS_13[0]) * x);
    evens_1.scaled_add(S::from_real(PADE_COEFFS_13[2]), a_2);
    evens_1.scaled_add(S::from_real(PADE_COEFFS_13[4]), a_4);
    evens_1.scaled_add(S::from_real(PADE_COEFFS_13[6]), a_6);

    let mut evens_2 = a_2.clone();
    evens_2.mapv_inplace(|x| S::from_real(PADE_COEFFS_13[8]) * x);
    evens_2.scaled_add(S::from_real(PADE_COEFFS_13[10]), a_4);
    evens_2.scaled_add(S::from_real(PADE_COEFFS_13[12]), a_6);
    let evens = evens_2.dot(a_6) + &evens_1;

    let mut odds_1: Array2<S> = Array::eye(a_1.nrows());
    odds_1.mapv_inplace(|x| S::from_real(PADE_COEFFS_13[1]) * x);
    odds_1.scaled_add(S::from_real(PADE_COEFFS_13[3]), a_2);
    odds_1.scaled_add(S::from_real(PADE_COEFFS_13[5]), a_4);
    odds_1.scaled_add(S::from_real(PADE_COEFFS_13[7]), a_6);

    let mut odds_2 = a_2.clone();
    odds_2.mapv_inplace(|x| S::from_real(PADE_COEFFS_13[9]) * x);
    odds_2.scaled_add(S::from_real(PADE_COEFFS_13[11]), a_4);
    odds_2.scaled_add(S::from_real(PADE_COEFFS_13[13]), a_6);
    odds_2 = odds_2.dot(a_6);

    let mut odds = (&odds_1 + &odds_2).dot(a_1);
    odds.mapv_inplace(|x| -x);
    let inverted: Array2<S> = (&odds + &evens).inv().unwrap();
    odds.mapv_inplace(|x| -x);
    inverted.dot(&(odds + evens))
}

fn power_abs_norm<S>(input_matrix: &Array2<S>, p: usize) -> f64
where
    S: Scalar<Real = f64>,
{
    let mut v = Array1::<f64>::ones((input_matrix.ncols()).f());
    let abs_matrix = input_matrix.t().map(|x| x.abs());
    for _ in 0..p {
        v.assign(&abs_matrix.dot(&v));
    }
    // return max col sum
    v.into_iter()
        .reduce(|x, y| if x > y { x } else { y })
        .unwrap()
}

// helper function used in Al-M & H. Name is unchanged for future reference.
fn ell<S: Scalar<Real = f64>>(a_matrix: &Array2<S>, m: u64) -> i32 {
    if a_matrix.is_square() == false {
        panic!("subroutine ell expected a square matrix.");
    }
    let p = 2 * m + 1;
    let a_one_norm = a_matrix.map(|x| x.abs()).opnorm_one().unwrap();

    if a_one_norm < f64::EPSILON * 2. {
        panic!("Subroutine ell encountered zero norm matrix.");
    }
    let powered_abs_norm = power_abs_norm(a_matrix, p as usize);
    let alpha = powered_abs_norm * pade_error_coefficient(m) / a_one_norm;
    let u = f64::EPSILON / 2.;
    let log2_alpha_div_u = f64::log2(alpha / u);
    let val = f64::ceil(log2_alpha_div_u / ((2 * m) as f64)) as i32;
    i32::max(val, 0)
}

/// Calculates the leading term of the error series for the [m/m] Pade approximation to exp(x).
fn pade_error_coefficient(m: u64) -> f64 {
    1.0 / (binomial(2 * m, m) * factorial(2 * m + 1))
}
/// Computes matrix exponential based on the scale-and-squaring algorithm by
pub fn expm<S: Scalar<Real = f64> + Lapack>(a_matrix: &Array2<S>) -> (Array2<S>, usize) {
    let mut a_2 = a_matrix.dot(a_matrix);
    let mut a_4 = a_2.dot(&a_2);
    let mut a_6 = a_2.dot(&a_4);
    let d4 = a_4.opnorm_one().unwrap().powf(1. / 4.);
    let d6 = a_6.opnorm_one().unwrap().powf(1. / 6.);
    // Note d6 should be an estimate and d4 an estimate
    let eta_1 = f64::max(d4, d6);
    if eta_1 < THETA_3 && ell(&a_matrix, 3) == 0 {
        return (pade_approximation_3(a_matrix, &a_2), 3);
    }
    // d4 should be exact here, d6 an estimate
    let eta_2 = f64::max(d4, d6);
    if eta_2 < THETA_5 && ell(&a_matrix, 5) == 0 {
        return (pade_approximation_5(a_matrix, &a_2, &a_4), 5);
    }
    let a_8 = a_4.dot(&a_4);
    let d8 = a_8.opnorm_one().unwrap().powf(1. / 8.);
    let eta_3 = f64::max(d6, d8);
    if eta_3 < THETA_7 && ell(&a_matrix, 7) == 0 {
        return (pade_approximation_7(a_matrix, &a_2, &a_4, &a_6), 7);
    }
    if eta_3 < THETA_9 && ell(&a_matrix, 9) == 0 {
        return (pade_approximation_9(a_matrix, &a_2, &a_4, &a_6, &a_8), 9);
    }
    let a_10 = a_2.dot(&a_8);
    let eta_4 = f64::max(d8, a_10.opnorm_one().unwrap());
    let eta_5 = f64::min(eta_3, eta_4);

    let mut s = f64::max(0., (eta_5 / THETA_13).log2().ceil()) as i32;
    let mut a_scaled = a_matrix.clone();
    let mut scaler = S::from_real(2.).powi(-s);
    a_scaled.mapv_inplace(|x| x * scaler);
    s += ell(&a_scaled, 13);

    a_scaled.assign(a_matrix);
    scaler = S::from_real(2.).powi(-s);
    a_scaled.mapv_inplace(|x| x * scaler);

    a_2.mapv_inplace(|x| x * scaler.powi(2));
    a_4.mapv_inplace(|x| x * scaler.powi(4));
    a_6.mapv_inplace(|x| x * scaler.powi(6));

    let mut output = pade_approximation_13(&a_scaled, &a_2, &a_4, &a_6);
    for _ in 0..s {
        output = output.dot(&output);
    }
    (output, 13)
}

mod tests {
    use crate::{
        expm::{
            pade_approximation_13, pade_approximation_3, pade_approximation_5,
            pade_approximation_7, pade_approximation_9,
        },
        Eig, OperationNorm, SVD,
    };
    use ndarray::{linalg::Dot, *};
    use num_complex::{Complex, Complex32 as c32, Complex64 as c64, ComplexFloat};
    use rand::Rng;
    use std::{collections::HashMap, fs::File, io::Read, str::FromStr};

    use super::expm;

    // 50 -> 5x worse error each entry than scipy
    // 100 -> 7.3x worse error each entry than scipy
    // 200 -> broken
    #[test]
    fn random_matrix_ensemble() {
        let mut rng = rand::thread_rng();
        let n = 200;
        let samps = 10;
        let mut results = Vec::new();
        let mut avg_entry_error = Vec::new();
        // Used to control what pade approximation is most likely to be used.
        // the smaller the norm the lower the degree used.
        let scale = 1.;
        for _ in 0..samps {
            // Sample a completely random matrix.
            let mut matrix: Array2<c64> = Array2::<c64>::ones((n, n).f());
            matrix.mapv_inplace(|_| c64::new(rng.gen::<f64>() * 1., rng.gen::<f64>() * 1.));

            // Make m positive semidefinite so it has orthonormal eigenvecs.
            matrix = matrix.dot(&matrix.t().map(|x| x.conj()));
            let (mut eigs, vecs) = matrix.eig().unwrap();
            let adjoint_vecs = vecs.t().clone().mapv(|x| x.conj());

            // Generate new random eigenvalues (complex, previously m had real eigenvals)
            // and a new matrix m
            eigs.mapv_inplace(|_| scale * c64::new(rng.gen::<f64>(), rng.gen::<f64>()));
            let new_matrix = vecs.dot(&Array2::from_diag(&eigs)).dot(&adjoint_vecs);

            // compute the exponentiated matrix by exponentiating the eigenvalues
            // and doing V e^Lambda V^\dagger
            eigs.mapv_inplace(|x| x.exp());
            let eigen_expm = vecs.dot(&Array2::from_diag(&eigs)).dot(&adjoint_vecs);

            // Compute the expm routine, compute error metrics for this sample
            let (expm_comp, deg) = expm(&new_matrix);
            let diff = &expm_comp - &eigen_expm;
            avg_entry_error.push({
                let tot = diff.map(|x| x.abs()).into_iter().sum::<f64>();
                tot / (n * n) as f64
            });
            results.push(diff.opnorm_one().unwrap());
        }

        // compute averages
        let avg: f64 = results.iter().sum::<f64>() / results.len() as f64;
        let avg_entry_diff = avg_entry_error.iter().sum::<f64>() / avg_entry_error.len() as f64;
        let std: f64 = f64::powf(
            results.iter().map(|x| f64::powi(x - avg, 2)).sum::<f64>() / (results.len() - 1) as f64,
            0.5,
        );
        println!("collected {:} samples.", results.len());
        println!("diff norm: {:} +- ({:})", avg, std);
        println!(
            "average entry error over epsilon: {:}",
            avg_entry_diff / f64::EPSILON
        );
        println!("avg over epsilon: {:.2}", avg / f64::EPSILON);
        println!("std over epsilon: {:.2}", std / f64::EPSILON);
    }

    #[test]
    fn test_pauli_rotation() {
        let mut results = Vec::new();
        let mut d3 = 0;
        let mut d5 = 0;
        let mut d7 = 0;
        let mut d9 = 0;
        let mut d13 = 0;
        let mut rng = rand::thread_rng();
        let num_samples = 10;
        for _ in 0..num_samples {
            let theta: c64 = c64::from_polar(2. * std::f64::consts::PI * rng.gen::<f64>(), 0.);
            let pauli_y: Array2<c64> = array![
                [c64::new(0., 0.), c64::new(0., -1.)],
                [c64::new(0., 1.), c64::new(0., 0.)],
            ];
            let x = c64::new(0., 1.) * theta * pauli_y.clone();
            let actual = c64::cos(theta / 2.) * Array2::<c64>::eye(x.nrows())
                - c64::sin(theta / 2.) * c64::new(0., 1.) * &pauli_y;
            let (computed, deg) = expm(&(c64::new(0., -0.5) * theta * pauli_y));
            match deg {
                3 => d3 += 1,
                5 => d5 += 1,
                7 => d7 += 1,
                9 => d9 += 1,
                13 => d13 += 1,
                _ => {}
            }
            let diff = (actual - computed).map(|x| x.abs());
            let diff_norm = diff.opnorm_one().unwrap();
            results.push(diff_norm);
        }
        let avg: f64 = results.iter().sum::<f64>() / results.len() as f64;
        let std: f64 = f64::powf(
            results.iter().map(|x| f64::powi(x - avg, 2)).sum::<f64>() / (results.len() - 1) as f64,
            0.5,
        );
        println!("collected {:} samples.", results.len());
        println!("diff norm: {:} +- ({:})", avg, std);
        println!("avg over epsilon: {:.2}", avg / f64::EPSILON);
        println!("std over epsilon: {:.2}", std / f64::EPSILON);
        println!("degree percentages: \n3  - {:.4}, \n5  - {:.4}, \n7  - {:.4}, \n9  - {:.4}, \n13 - {:.4}",
        d3 as f64 / num_samples as f64,
        d5 as f64 / num_samples as f64,
        d7 as f64 / num_samples as f64,
        d9 as f64 / num_samples as f64,
        d13 as f64 / num_samples as f64);
        // println!("results: {:?}", results);
    }
    #[test]
    fn test_pade_approximants() {
        let mat: Array2<f64> =
            10. as f64 * array![[0.1, 0.2, 0.3], [0.2, 0.1, 0.5], [0.11, 0.22, 0.32]];
        let mut output_3: Array2<f64> = Array::zeros((3, 3).f());
        let mut output_5: Array2<f64> = Array::zeros((3, 3).f());
        let mut output_7: Array2<f64> = Array::zeros((3, 3).f());
        let mut output_9: Array2<f64> = Array::zeros((3, 3).f());
        let mut output_13: Array2<f64> = Array::zeros((3, 3).f());
        fn compute_pade_diff_error(output: &Array2<f64>, expected: &Array2<f64>) -> f64 {
            let mut tot = 0.0;
            for row_ix in 0..output.nrows() {
                for col_ix in 0..output.ncols() {
                    tot += (output[[row_ix, col_ix]] - expected[[row_ix, col_ix]]).abs();
                }
            }
            tot
        }
        // pade_approximation_3(&mut mp, &mut output_3);
        // pade_approximation_5(&mut mp, &mut output_5);
        // pade_approximation_7(&mut mp, &mut output_7);
        // pade_approximation_9(&mut mp, &mut output_9);/
        // pade_approximation_13(&mut mp, &mut output_13);
        let expected = array![
            [157.766, 217.949, 432.144],
            [200.674, 278.691, 552.725],
            [169.969, 236.289, 469.437]
        ];
        println!("output?");
        println!(
            "3 norm error: {:}",
            compute_pade_diff_error(&output_3, &expected)
        );
        println!(
            "5 norm error: {:}",
            compute_pade_diff_error(&output_5, &expected)
        );
        println!(
            "7 norm error: {:}",
            compute_pade_diff_error(&output_7, &expected)
        );
        println!(
            "9 norm error: {:}",
            compute_pade_diff_error(&output_9, &expected)
        );
        println!(
            "13 norm error: {:}",
            compute_pade_diff_error(&output_13, &expected)
        );
    }
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
