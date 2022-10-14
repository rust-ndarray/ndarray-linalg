use crate::{types, Inverse};
use ndarray::prelude::*;
use statrs::function::factorial::{binomial, factorial};

// These constants are hard-coded from Al-Mohy & Higham
const THETA_3: f64 = 1.495585217958292e-2;
const THETA_5: f64 = 2.539398330063230e-1;
const THETA_7: f64 = 9.504178996162932e-1;
const THETA_9: f64 = 2.097847961257068e0;
const THETA_13: f64 = 4.25; // Alg 5.1

// this is pure laziness aka "ergonomics"
const THETA_MAP: [f64; 14] = [
    0., 0., 0., THETA_3, 0., THETA_5, 0., THETA_7, 0., THETA_9, 0., 0., 0., THETA_13,
];

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

// helper function used in Al-M & H. Name is unchanged for future reference.
fn ell(A: Array2<f64>, m: u64) -> i32 {
    1
}

/// Calculates the leading term of the error series for the [m/m] Pade approximation to exp(x).
fn pade_error_coefficient(m: u64) -> f64 {
    1.0 / (binomial(2 * m, m) * factorial(2 * m + 1))
}

fn pade_approximation_3(input: &Array2<f64>, output: &mut Array2<f64>) {
    let input_2 = input.dot(input);
    let evens = PADE_COEFFS_3[0] * Array::eye(input.nrows()) + PADE_COEFFS_3[2] * input_2.clone();
    let odds =
        (PADE_COEFFS_3[1] * Array::eye(input.nrows()) + PADE_COEFFS_3[3] * input_2).dot(input);
    let inverted = (-1. * odds.clone() + evens.clone()).inv().unwrap();
    output.assign(&inverted.dot(&(odds + evens)));
}

fn pade_approximation_5(input: &Array2<f64>, output: &mut Array2<f64>) {
    let input_2 = input.dot(input);
    let input_4 = input_2.dot(&input_2);
    let evens = PADE_COEFFS_5[0] * Array::eye(input.nrows())
        + PADE_COEFFS_5[2] * input_2.clone()
        + PADE_COEFFS_5[4] * input_4.clone();
    let odds = (PADE_COEFFS_5[1] * Array::eye(input.nrows())
        + PADE_COEFFS_5[3] * input_2
        + PADE_COEFFS_5[5] * input_4)
        .dot(input);
    let inverted = (-1. * odds.clone() + evens.clone()).inv().unwrap();
    output.assign(&inverted.dot(&(odds + evens)));
}

fn pade_approximation_7(input: &Array2<f64>, output: &mut Array2<f64>) {
    let input_2 = input.dot(input);
    let input_4 = input_2.dot(&input_2);
    let input_6 = input_2.dot(&input_4);
    let evens = PADE_COEFFS_7[0] * Array::eye(input.nrows())
        + PADE_COEFFS_7[2] * input_2.clone()
        + PADE_COEFFS_7[4] * input_4.clone()
        + PADE_COEFFS_7[6] * input_6.clone();
    let odds = (PADE_COEFFS_7[1] * Array::eye(input.nrows())
        + PADE_COEFFS_7[3] * input_2
        + PADE_COEFFS_7[5] * input_4
        + PADE_COEFFS_7[7] * input_6)
        .dot(input);
    let inverted = (-1. * odds.clone() + evens.clone()).inv().unwrap();
    output.assign(&inverted.dot(&(odds + evens)));
}
fn pade_approximation_9(input: &Array2<f64>, output: &mut Array2<f64>) {
    let input_2 = input.dot(input);
    let input_4 = input_2.dot(&input_2);
    let input_6 = input_2.dot(&input_4);
    let input_8 = input_4.dot(&input_4);
    let evens = PADE_COEFFS_9[0] * Array::eye(input.nrows())
        + PADE_COEFFS_9[2] * input_2.clone()
        + PADE_COEFFS_9[4] * input_4.clone()
        + PADE_COEFFS_9[6] * input_6.clone()
        + PADE_COEFFS_9[8] * input_8.clone();
    let odds = (PADE_COEFFS_9[1] * Array::eye(input.nrows())
        + PADE_COEFFS_9[3] * input_2
        + PADE_COEFFS_9[5] * input_4
        + PADE_COEFFS_9[7] * input_6
        + PADE_COEFFS_9[9] * input_8)
        .dot(input);
    let inverted = (-1. * odds.clone() + evens.clone()).inv().unwrap();
    output.assign(&inverted.dot(&(odds + evens)));
}
fn pade_approximation_13(input: &Array2<f64>, output: &mut Array2<f64>) {
    let input_2 = input.dot(input);
    let input_4 = input_2.dot(&input_2);
    let input_6 = input_2.dot(&input_4);
    let input_8 = input_4.dot(&input_4);
    let evens_1 = PADE_COEFFS_13[0] * Array::eye(input.nrows())
        + PADE_COEFFS_13[2] * input_2.clone()
        + PADE_COEFFS_13[4] * input_4.clone()
        + PADE_COEFFS_13[6] * input_6.clone();
    let evens_2 = (PADE_COEFFS_13[8] * input_2.clone()
        + PADE_COEFFS_13[10] * input_4.clone()
        + PADE_COEFFS_13[12] * input_6.clone())
    .dot(&input_6);
    let evens = evens_1 + evens_2;
    let odds_1 = PADE_COEFFS_13[1] * Array::eye(input.nrows())
        + PADE_COEFFS_13[3] * input_2.clone()
        + PADE_COEFFS_13[5] * input_4.clone()
        + PADE_COEFFS_13[7] * input_6.clone();
    let odds_2 = (PADE_COEFFS_13[9] * input_2
        + PADE_COEFFS_13[11] * input_4
        + PADE_COEFFS_13[13] * input_6.clone())
    .dot(&input_6);
    let odds = (odds_1 + odds_2).dot(input);
    let inverted = (-1. * odds.clone() + evens.clone()).inv().unwrap();
    output.assign(&inverted.dot(&(odds + evens)));
}

fn pade_approximation(input: &Array2<f64>, output: &mut Array2<f64>, degree: &usize) {
    match *degree {
        3 => pade_approximation_3(input, output),
        5 => pade_approximation_5(input, output),
        7 => pade_approximation_7(input, output),
        9 => pade_approximation_9(input, output),
        13 => pade_approximation_13(input, output),
        _ => {
            println!("Undefined pade approximant order.")
        }
    }
}

fn old_expm(matrix: &Array2<f64>) -> Array2<f64> {
    let mut ret: Array2<f64> = ndarray::Array2::<f64>::zeros((matrix.nrows(), matrix.ncols()));
    for m in vec![3, 5, 7, 9].iter() {
        if super::normest1::normest(matrix, 4, 4) <= THETA_MAP[*m] {
            pade_approximation(matrix, &mut ret, m);
            return ret;
        }
    }
    ret
}
/// This function is based on the scale-and-squaring algorithm by
pub fn expm(A: Array2<i32>) -> Array2<i32> {
    arr2(&[[2]])
}

mod tests {
    use crate::expm::{
        pade_approximation_13, pade_approximation_3, pade_approximation_5, pade_approximation_7,
        pade_approximation_9,
    };
    use ndarray::{linalg::Dot, *};

    #[test]
    fn test_pade_approximants() {
        let mat = 10. * array![[0.1, 0.2, 0.3], [0.2, 0.1, 0.5], [0.11, 0.22, 0.32]];
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
        pade_approximation_3(&mat, &mut output_3);
        pade_approximation_5(&mat, &mut output_5);
        pade_approximation_7(&mat, &mut output_7);
        pade_approximation_9(&mat, &mut output_9);
        pade_approximation_13(&mat, &mut output_13);
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
