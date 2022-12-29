use crate::{Inverse, OperationNorm};
use cauchy::Scalar;
use lax::Lapack;
use ndarray::prelude::*;
extern crate statrs;
use statrs::{
    function::factorial::{binomial, factorial},
};
use crate::error::{Result, LinalgError};

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

fn pade_approximation_3<S: Scalar<Real = f64> + Lapack>(
    a_1: &Array2<S>,
    a_2: &Array2<S>,
) -> Result<Array2<S>> {
    let mut evens: Array2<S> = Array2::<S>::eye(a_1.nrows());
    // evens.mapv_inplace(|x| x * S::from_real(PADE_COEFFS_3[0]));
    evens.scaled_add(S::from_real(PADE_COEFFS_3[2]), a_2);

    let mut odds: Array2<S> = Array2::<S>::eye(a_1.nrows());
    odds.mapv_inplace(|x| x * S::from_real(PADE_COEFFS_3[1]));
    odds.scaled_add(S::from_real(PADE_COEFFS_3[3]), a_2);
    odds = odds.dot(a_1);

    odds.mapv_inplace(|x| -x);
    let inverted = (&odds + &evens).inv()?;
    odds.mapv_inplace(|x| -x);
    Ok(inverted.dot(&(odds + evens)))
}

fn pade_approximation_5<S: Scalar<Real = f64> + Lapack>(
    a_1: &Array2<S>,
    a_2: &Array2<S>,
    a_4: &Array2<S>,
) -> Result<Array2<S>> {
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
    let inverted = (&odds + &evens).inv()?;
    odds.mapv_inplace(|x| -x);
    Ok(inverted.dot(&(odds + evens)))
}

fn pade_approximation_7<S: Scalar<Real = f64> + Lapack>(
    a_1: &Array2<S>,
    a_2: &Array2<S>,
    a_4: &Array2<S>,
    a_6: &Array2<S>,
) -> Result<Array2<S>> {
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
    let inverted = (&odds + &evens).inv()?;
    odds.mapv_inplace(|x| -x);
    Ok(inverted.dot(&(odds + evens)))
}

fn pade_approximation_9<S: Scalar<Real = f64> + Lapack>(
    a_1: &Array2<S>,
    a_2: &Array2<S>,
    a_4: &Array2<S>,
    a_6: &Array2<S>,
    a_8: &Array2<S>,
) -> Result<Array2<S>> {
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
    let inverted: Array2<S> = (&odds + &evens).inv()?;
    odds.mapv_inplace(|x| -x);
    Ok(inverted.dot(&(odds + evens)))
}

// Note: all input matrices should be scaled in the main expm
// function. 
fn pade_approximation_13<S: Scalar<Real = f64> + Lapack>(
    a_1: &Array2<S>,
    a_2: &Array2<S>,
    a_4: &Array2<S>,
    a_6: &Array2<S>,
) -> Result<Array2<S>> {
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
    let inverted: Array2<S> = (&odds + &evens).inv()?;
    odds.mapv_inplace(|x| -x);
    Ok(inverted.dot(&(odds + evens)))
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

/// helper function used in Al-Mohy & Higham. Name is unchanged for future reference.
fn ell<S: Scalar<Real = f64>>(a_matrix: &Array2<S>, m: u64) -> Result<i32> {
    if a_matrix.is_square() == false {
        return Err(LinalgError::NotSquare { rows: a_matrix.nrows() as i32, cols: a_matrix.ncols() as i32 });
    }
    let p = 2 * m + 1;
    let a_one_norm = a_matrix.map(|x| x.abs()).opnorm_one()?;

    let powered_abs_norm = power_abs_norm(a_matrix, p as usize);
    let alpha = powered_abs_norm * pade_error_coefficient(m) / a_one_norm;
    let u = f64::EPSILON / 2.;
    let log2_alpha_div_u = f64::log2(alpha / u);
    let val = f64::ceil(log2_alpha_div_u / ((2 * m) as f64)) as i32;
    Ok(i32::max(val, 0))
}

/// Calculates the leading term of the error series for the [m/m] Pade approximation to exp(x).
fn pade_error_coefficient(m: u64) -> f64 {
    1.0 / (binomial(2 * m, m) * factorial(2 * m + 1))
}

/// ## Matrix Exponentiation
/// Computes matrix exponential based on the scaling-and-squaring algorithm by Al-Mohy and Higham.
/// Currently restricted to matrices with entries that are either f64 or Complex64. 64 bit precision is required 
/// due to error calculations in Al-Mohy and Higham. Utilizes Lapack
/// calls so entries must satisfy LAPACK trait bounds. 
/// 
/// ### Caveats
/// Currently confirmed accurate to f64 precision up to 1024x1024 sparse matrices. Dense matrices
/// have been observed with a worse average entry error, up to 100x100 matrices should be close 
/// enough to f64 precision for vast majority of numeric purposes.
pub fn expm<S: Scalar<Real = f64> + Lapack>(a_matrix: &Array2<S>) -> Result<Array2<S>> {
    let mut a_2 = a_matrix.dot(a_matrix);
    let mut a_4 = a_2.dot(&a_2);
    let mut a_6 = a_2.dot(&a_4);
    let d4 = a_4.opnorm_one()?.powf(1. / 4.);
    let d6 = a_6.opnorm_one()?.powf(1. / 6.);
    // Note d6 should be an estimate and d4 an estimate
    let eta_1 = f64::max(d4, d6);
    if eta_1 < THETA_3 && ell(&a_matrix, 3)? == 0 {
        return pade_approximation_3(a_matrix, &a_2);
    }
    // d4 should be exact here, d6 an estimate
    let eta_2 = f64::max(d4, d6);
    if eta_2 < THETA_5 && ell(&a_matrix, 5)? == 0 {
        return pade_approximation_5(a_matrix, &a_2, &a_4);
    }
    let a_8 = a_4.dot(&a_4);
    let d8 = a_8.opnorm_one().unwrap().powf(1. / 8.);
    let eta_3 = f64::max(d6, d8);
    if eta_3 < THETA_7 && ell(&a_matrix, 7)? == 0 {
        return pade_approximation_7(a_matrix, &a_2, &a_4, &a_6);
    }
    if eta_3 < THETA_9 && ell(&a_matrix, 9)? == 0 {
        return pade_approximation_9(a_matrix, &a_2, &a_4, &a_6, &a_8);
    }
    let a_10 = a_2.dot(&a_8);
    let eta_4 = f64::max(d8, a_10.opnorm_one()?);
    let eta_5 = f64::min(eta_3, eta_4);

    let mut s = f64::max(0., (eta_5 / THETA_13).log2().ceil()) as i32;
    let mut a_scaled = a_matrix.clone();
    let mut scaler = S::from_real(2.).powi(-s);
    a_scaled.mapv_inplace(|x| x * scaler);
    s += ell(&a_scaled, 13)?;

    a_scaled.assign(a_matrix);
    scaler = S::from_real(2.).powi(-s);
    a_scaled.mapv_inplace(|x| x * scaler);
    a_2.mapv_inplace(|x| x * scaler.powi(2));
    a_4.mapv_inplace(|x| x * scaler.powi(4));
    a_6.mapv_inplace(|x| x * scaler.powi(6));

    let mut output = pade_approximation_13(&a_scaled, &a_2, &a_4, &a_6)?;
    for _ in 0..s {
        output = output.dot(&output);
    }
    Ok(output)
}