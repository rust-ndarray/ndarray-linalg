use std::ops::MulAssign;

use crate::{normest1::normest, types, Inverse, OperationNorm};
use cauchy::{Scalar};
use lax::{Lapack, NormType};
use ndarray::{linalg::Dot, prelude::*};
use num_complex::{Complex, Complex32 as c32, Complex64 as c64};
use num_traits::{real::Real, Pow};
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

// this is pure laziness aka "ergonomics"
const THETA_MAP: [f64; 14] = [
    0., 0., 0., THETA_3, 0., THETA_5, 0., THETA_7, 0., THETA_9, 0., 0., 0., THETA_13,
];

// The Pade Coefficients for the numerator of the diagonal approximation to Exp[x]. Computed via Mathematica/WolframAlpha.
// Note that the denominator has the same coefficients but odd powers have an opposite sign.
// Coefficients are also stored via the power of x, so for example the numerator would look like
// x^0 * PADE_COEFF_M[0] + x^1 * PADE_COEFF_M[1] + ... + x^m * PADE_COEFF_M[M]
const PADE_COEFFS_3: [f64; 4] = [1., 0.5, 0.1, 1. / 120.];
// const PADE_COEFFS_3:[f64; 4] = [1., 12., 60., 120.];

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

    if a_one_norm < f64::EPSILON * 2.0 {
        panic!("Subroutine ell encountered zero norm matrix.");
    }
    let powered_abs_norm = power_abs_norm(a_matrix, p as usize);
    let alpha = powered_abs_norm * pade_error_coefficient(m) / a_one_norm;
    let u = f64::EPSILON / 2.;
    let log2_alpha_div_u = f64::log2(alpha / u);
    let val = f64::ceil(log2_alpha_div_u / ((2 * m) as f64)).round() as i32;
    i32::max(val, 0)
}

/// Calculates the leading term of the error series for the [m/m] Pade approximation to exp(x).
fn pade_error_coefficient(m: u64) -> f64 {
    1.0 / (binomial(2 * m, m) * factorial(2 * m + 1))
}

fn pade_approximation_3<S: Scalar<Real = f64> + Lapack>(
    mp: &mut MatrixPowers<S>,
    output: &mut Array2<S>,
) {
    let mut evens: Array2<S> = Array2::<S>::eye(mp.get(1).nrows());
    // evens.mapv_inplace(|x| x * S::from_real(PADE_COEFFS_3[0]));
    evens.scaled_add(S::from_real(PADE_COEFFS_3[2]), mp.get(2));

    let mut odds: Array2<S> = Array2::<S>::eye(mp.get(1).nrows());
    odds.mapv_inplace(|x| x * S::from_real(PADE_COEFFS_3[1]));
    odds.scaled_add(S::from_real(PADE_COEFFS_3[3]), mp.get(2));
    odds = odds.dot(mp.get(1));

    odds.mapv_inplace(|x| -x);
    let inverted = (&odds + &evens).inv().unwrap();
    odds.mapv_inplace(|x| -x);
    output.assign(&inverted.dot(&(odds + evens)));
}

fn pade_approximation_5<S: Scalar<Real = f64> + Lapack>(
    mp: &mut MatrixPowers<S>,
    output: &mut Array2<S>,
) {
    let mut evens: Array2<S> = Array2::<S>::eye(mp.get(1).nrows());
    // evens.mapv_inplace(|x| S::from_real(PADE_COEFFS_5[0]) * x);
    for m in 1..=2 {
        evens.scaled_add(S::from_real(PADE_COEFFS_5[2 * m]), mp.get(2 * m));
    }
    let mut odds: Array2<S> = Array::eye(mp.get(1).nrows());
    odds.mapv_inplace(|x| S::from_real(PADE_COEFFS_5[1]) * x);
    for m in 1..=2 {
        odds.scaled_add(S::from_real(PADE_COEFFS_5[2 * m + 1]), mp.get(2 * m));
    }
    odds = odds.dot(mp.get(1));
    odds.mapv_inplace(|x| -x);
    let inverted = (&odds + &evens).inv().unwrap();
    odds.mapv_inplace(|x| -x);
    output.assign(&inverted.dot(&(odds + evens)));
}

fn pade_approximation_7<S: Scalar<Real = f64> + Lapack>(
    mp: &mut MatrixPowers<S>,
    output: &mut Array2<S>,
) {
    let mut evens: Array2<S> = Array::eye(mp.get(1).nrows());
    // evens.mapv_inplace(|x| S::from_real(PADE_COEFFS_7[0]) * x);
    for m in 1..=3 {
        evens.scaled_add(S::from_real(PADE_COEFFS_7[2 * m]), mp.get(2 * m));
    }
    let mut odds: Array2<S> = Array::eye(mp.get(1).nrows());
    odds.mapv_inplace(|x| S::from_real(PADE_COEFFS_7[1]) * x);
    for m in 1..=3 {
        odds.scaled_add(S::from_real(PADE_COEFFS_7[2 * m + 1]), mp.get(2 * m));
    }
    odds = odds.dot(mp.get(1));
    odds.mapv_inplace(|x| -x);
    let inverted = (&odds + &evens).inv().unwrap();
    odds.mapv_inplace(|x| -x);
    output.assign(&inverted.dot(&(odds + evens)));
}
fn pade_approximation_9<S: Scalar<Real = f64> + Lapack>(
    mp: &mut MatrixPowers<S>,
    output: &mut Array2<S>,
) {
    let mut evens: Array2<S> = Array::eye(mp.get(1).nrows());
    // evens.mapv_inplace(|x| S::from_real(PADE_COEFFS_9[0]) * x);
    for m in 1..=4 {
        evens.scaled_add(S::from_real(PADE_COEFFS_9[2 * m]), mp.get(2 * m));
        let delta = mp.get(2 * m).map(|x| S::from_real(PADE_COEFFS_9[2 * m]) * *x);
    }
    let mut odds: Array2<S> = Array::eye(mp.get(1).nrows());
    odds.mapv_inplace(|x| S::from_real(PADE_COEFFS_9[1]) * x);
    for m in 1..=4 {
        odds.scaled_add(S::from_real(PADE_COEFFS_9[2 * m + 1]), mp.get(2 * m));
    }
    odds = odds.dot(mp.get(1));
    odds.mapv_inplace(|x| -x);
    let inverted: Array2<S> = (&odds + &evens).inv().unwrap();
    odds.mapv_inplace(|x| -x);
    output.assign(&inverted.dot(&(odds + evens)));
}

// TODO: scale powers by appropriate value of s.
fn pade_approximation_13<S: Scalar<Real = f64> + Lapack>(
    mp: &mut MatrixPowers<S>,
    output: &mut Array2<S>,
) {
    // note this may have unnecessary allocations.
    let mut evens_1: Array2<S> = Array::eye(mp.get(1).nrows());
    evens_1.mapv_inplace(|x| S::from_real(PADE_COEFFS_13[0]) * x);
    for m in 1..=3 {
        evens_1.scaled_add(S::from_real(PADE_COEFFS_13[2 * m]), mp.get(2 * m));
    }
    let mut evens_2 = mp.get(2).clone();
    evens_2.mapv_inplace(|x| S::from_real(PADE_COEFFS_13[8]) * x);
    evens_2.scaled_add(S::from_real(PADE_COEFFS_13[10]), mp.get(4));
    evens_2.scaled_add(S::from_real(PADE_COEFFS_13[12]), mp.get(6));
    let evens = evens_2.dot(mp.get(6)) + &evens_1;
    let mut odds_1: Array2<S> = Array::eye(mp.get(1).nrows());
    odds_1.mapv_inplace(|x| S::from_real(PADE_COEFFS_13[1]) * x);
    for m in 1..=3 {
        odds_1.scaled_add(S::from_real(PADE_COEFFS_13[2 * m + 1]), mp.get(2 * m));
    }
    let mut odds_2 = mp.get(2).clone();
    odds_2.mapv_inplace(|x| S::from_real(PADE_COEFFS_13[9]) * x);
    odds_2.scaled_add(S::from_real(PADE_COEFFS_13[11]), mp.get(4));
    odds_2.scaled_add(S::from_real(PADE_COEFFS_13[13]), mp.get(6));
    odds_2 = odds_2.dot(mp.get(6));
    let mut odds = (odds_1 + odds_2).dot(mp.get(1));
    odds.mapv_inplace(|x| -x);
    let inverted: Array2<S> = (&odds + &evens).inv().unwrap();
    odds.mapv_inplace(|x| -x);
    output.assign(&inverted.dot(&(odds + evens)));
}

/// Helper struct to ensure that the power of the input matrix is only computed once.
#[derive(Debug)]
struct MatrixPowers<S: Scalar + Lapack> {
    pub input_1: Option<Array2<S>>,
    pub input_2: Option<Array2<S>>,
    pub input_3: Option<Array2<S>>,
    pub input_4: Option<Array2<S>>,
    pub input_5: Option<Array2<S>>,
    pub input_6: Option<Array2<S>>,
    pub input_7: Option<Array2<S>>,
    pub input_8: Option<Array2<S>>,
    pub input_9: Option<Array2<S>>,
    pub input_10: Option<Array2<S>>,
    pub input_11: Option<Array2<S>>,
    pub input_12: Option<Array2<S>>,
    pub input_13: Option<Array2<S>>,
    pub num_matprods: usize,
}
impl<S: Scalar + Lapack> MatrixPowers<S> {
    pub fn new(input: Array2<S>) -> Self {
        MatrixPowers {
            input_1: Some(input),
            input_2: None,
            input_3: None,
            input_4: None,
            input_5: None,
            input_6: None,
            input_7: None,
            input_8: None,
            input_9: None,
            input_10: None,
            input_11: None,
            input_12: None,
            input_13: None,
            num_matprods: 0,
        }
    }
    pub fn scale(&mut self, s: i32) {
        let scale_factor = S::from_i32(2).unwrap().pow(-S::from_i32(s).unwrap());
        let mut scaler = scale_factor.clone();
        if let Some(i1) = &mut self.input_1 {
            i1.mapv_inplace(|x| x * scaler);
        }
        scaler *= scale_factor;
        if let Some(i2) = &mut self.input_2 {
            i2.mapv_inplace(|x| x * scaler);
        }
        scaler *= scale_factor;
        if let Some(i3) = &mut self.input_3 {
            i3.mapv_inplace(|x| x * scaler);
        }
        scaler *= scale_factor;
        if let Some(i4) = &mut self.input_4 {
            i4.mapv_inplace(|x| x * scaler);
        }
        scaler *= scale_factor;
        if let Some(i5) = &mut self.input_5 {
            i5.mapv_inplace(|x| x * scaler);
        }
        scaler *= scale_factor;
        if let Some(i6) = &mut self.input_6 {
            i6.mapv_inplace(|x| x * scaler);
        }
    }
    fn compute2(&mut self) {
        if let Some(input_1) = self.input_1.clone() {
            self.input_2 = Some(input_1.dot(&input_1));
            self.num_matprods += 1;
        }
    }
    fn compute3(&mut self) {
        match &self.input_2 {
            Some(i2) => {
                self.input_3 = Some(self.input_1.as_ref().unwrap().dot(i2));
                self.num_matprods += 1;
            }
            None => {
                // after calling self.compute2() then self.input_2 will match to Some(_)
                // so just recurse.
                self.compute2();
                self.compute3();
            }
        }
    }
    fn compute4(&mut self) {
        match &self.input_2 {
            Some(i2) => {
                self.input_4 = Some(self.input_2.as_ref().unwrap().dot(i2));
                self.num_matprods += 1;
            }
            None => {
                self.compute2();
                self.compute4();
            }
        }
    }
    fn compute5(&mut self) {
        match &self.input_3 {
            Some(i3) => match &self.input_2 {
                Some(i2) => {
                    self.input_5 = Some(i2.dot(i3));
                    self.num_matprods += 1;
                }
                None => {
                    self.compute2();
                    self.compute5();
                }
            },
            None => {
                self.compute3();
                self.compute5();
            }
        };
    }
    fn compute6(&mut self) {
        match &self.input_4 {
            Some(i4) => {
                // If input_4 is computed then input_2 must be computed.
                self.input_6 = Some(i4.dot(self.input_2.as_ref().unwrap()));
                self.num_matprods += 1;
            }
            None => {
                match &self.input_3 {
                    Some(i3) => {
                        self.input_6 = Some(i3.dot(i3));
                        self.num_matprods += 1;
                    }
                    None => {
                        // We do not have 4 or 3 computed yet, so we will either have 2 or have to compute it.
                        // in that case easiest way is to first compute 4 and then revisit.
                        self.compute4();
                        self.compute6();
                    }
                }
            }
        };
    }
    fn compute7(&mut self) {
        match &self.input_6 {
            Some(i6) => {
                self.input_7 = Some(self.input_1.as_ref().unwrap().dot(i6));
                self.num_matprods += 1;
            },
            None => {
                match &self.input_5 {
                    Some(i5) => {
                        self.input_7 = Some(self.input_2.as_ref().unwrap().dot(i5));
                        self.num_matprods += 1;
                    },
                    None => {
                        match &self.input_4 {
                            Some(i4) => {
                                if let Some(i3) = &self.input_3 {
                                    self.input_7 = Some(i3.dot(i4));
                                    self.num_matprods += 1;
                                } else {
                                    self.compute3();
                                    self.compute7();
                                }
                            },
                            None => {
                                self.compute4();
                                self.compute7();
                            }
                        }
                    },
                }
            }
        }
    }
    fn compute8(&mut self) {
        match &self.input_4 {
            Some(i4) => {
                self.input_8 = Some(i4.dot(i4));
                self.num_matprods += 1;
            },
            None => {
                self.compute4();
                self.compute8();
            }
        }
    }
    fn compute10(&mut self) {
        match &self.input_5 {
            Some(i5) => {
                self.input_10 = Some(i5.dot(i5));
                self.num_matprods += 1;
            },
            None => {
                match &self.input_6 {
                    Some(i6) => {
                        self.input_10 = Some(i6.dot(self.input_4.as_ref().unwrap()));
                        self.num_matprods += 1;
                    },
                    None => {
                        if self.input_3.is_some() {
                            self.compute5();
                        } else {
                            self.compute6()
                        }
                        self.compute10();
                    }
                }
            }
        }
    }

    pub fn get(&mut self, m: usize) -> &Array2<S> {
        match m {
            1 => self.get1(),
            2 => self.get2(),
            3 => self.get3(),
            4 => self.get4(),
            5 => self.get5(),
            6 => self.get6(),
            7 => self.get7(),
            8 => self.get8(),
            // 9 => self.get9(),
            10 => self.get10(),
            // 11 => self.get11(),
            // 12 => self.get12(),
            // 13 => self.get13(),
            _ => {
                println!("I need:{:}", m);
                println!("This power of input matrix is not implemented. Returning input matrix.");
                self.input_1.as_ref().unwrap()
            }
        }
    }
    fn get1(&mut self) -> &Array2<S> {
        self.input_1.as_ref().unwrap()
    }
    fn get2(&mut self) -> &Array2<S> {
        match &self.input_2 {
            Some(mat) => self.input_2.as_ref().unwrap(),
            None => {
                self.compute2();
                self.input_2.as_ref().unwrap()
            }
        }
    }
    fn get3(&mut self) -> &Array2<S> {
        match &self.input_3 {
            Some(mat) => self.input_3.as_ref().unwrap(),
            None => {
                self.compute3();
                &self.input_3.as_ref().unwrap()
            }
        }
    }
    fn get4(&mut self) -> &Array2<S> {
        match &self.input_4 {
            Some(mat) => &self.input_4.as_ref().unwrap(),
            None => {
                self.compute4();
                &self.input_4.as_ref().unwrap()
            }
        }
    }
    fn get5(&mut self) -> &Array2<S> {
        match &self.input_5 {
            Some(mat) => &self.input_5.as_ref().unwrap(),
            None => {
                self.compute5();
                &self.input_5.as_ref().unwrap()
            }
        }
    }
    fn get6(&mut self) -> &Array2<S> {
        match &self.input_6 {
            Some(mat) => &self.input_6.as_ref().unwrap(),
            None => {
                self.compute6();
                &self.input_6.as_ref().unwrap()
            }
        }
    }
    fn get7(&mut self) -> &Array2<S> {
        match &self.input_7 {
            Some(mat) => &self.input_6.as_ref().unwrap(),
            None => {
                self.compute7();
                &self.input_6.as_ref().unwrap()
            }
        }
    }
    fn get8(&mut self) -> &Array2<S> {
        match &self.input_8 {
            Some(mat) => &self.input_8.as_ref().unwrap(),
            None => {
                self.compute8();
                &self.input_8.as_ref().unwrap()
            }
        }
    }
    fn get10(&mut self) -> &Array2<S> {
        match &self.input_10 {
            Some(mat) => &self.input_10.as_ref().unwrap(),
            None => {
                self.compute10();
                &self.input_10.as_ref().unwrap()
            }
        }
    }
}

fn pade_approximation<'a, S: Scalar<Real = f64> + Lapack>(
    mp: &'a mut MatrixPowers<S>,
    output: &mut Array2<S>,
    degree: usize,
) {
    match degree {
        3 => {
            pade_approximation_3(mp, output);
        }
        5 => {
            pade_approximation_5(mp, output);
        }
        7 => {
            pade_approximation_7(mp, output);
        }
        9 => {
            pade_approximation_9(mp, output);
        }
        13 => {
            pade_approximation_13(mp, output);
        }
        _ => {
            println!(
                "Undefined pade approximant order. Returning first order approximation to expm"
            );
            output.assign(&(Array2::<S>::eye(mp.get(1).nrows()) + mp.get(1)));
        }
    }
}

/// Computes matrix exponential based on the scale-and-squaring algorithm by
pub fn expm<S: Scalar<Real = f64> + Lapack>(a_matrix: &Array2<S>) -> (Array2<S>, usize) {
    let mut output = Array2::<S>::zeros((a_matrix.nrows(), a_matrix.ncols()).f());
    let mut mp = MatrixPowers::new(a_matrix.clone());
    let d4 = f64::powf(mp.get(4).opnorm_one().unwrap(), 1. / 4.);
    let d6 = f64::powf((*mp.get(6)).opnorm_one().unwrap(), 1. / 6.);
    // Note d6 should be an estimate and d4 an estimate
    let eta_1 = f64::max(d4, d6);
    if eta_1 < THETA_3 && ell(&a_matrix, 3) == 0 {
        pade_approximation(&mut mp, &mut output, 3);
        return (output, 3);
    }
    // d4 should be exact here, d6 an estimate
    let eta_2 = f64::max(d4, d6);
    if eta_2 < THETA_5 && ell(&a_matrix, 5) == 0 {
        pade_approximation(&mut mp, &mut output, 5);
        return (output, 5);
    }
    let d8 = f64::powf(mp.get(8).opnorm_one().unwrap(), 1. / 8.);
    let eta_3 = f64::max(d6, d8);
    if eta_3 < THETA_7 && ell(&a_matrix, 7) == 0 {
        pade_approximation(&mut mp, &mut output, 7);
        return (output, 7);
    }
    if eta_3 < THETA_9 && ell(&a_matrix, 9) == 0 {
        pade_approximation(&mut mp, &mut output, 9);
        return (output, 9);
    }
    let eta_4 = f64::max(d8, mp.get(10).opnorm_one().unwrap());
    let eta_5 = f64::min(eta_3, eta_4);
    let mut s = f64::max(0., (eta_5 / THETA_13).log2().ceil()) as i32;
    let mut a_scaled = a_matrix.clone();
    a_scaled.mapv_inplace(|x| x * S::from_real(cauchy::Scalar::pow(2., -s as f64)));
    s += ell(&a_scaled, 13);
    mp.scale(s);
    pade_approximation(&mut mp, &mut output, 13);
    for _ in 0..s {
        output = output.dot(&output);
    }
    (output, 13)
}

mod tests {
    use std::{collections::HashMap, fs::File, io::Read, str::FromStr};

    use crate::{expm::{
        pade_approximation_13, pade_approximation_3, pade_approximation_5, pade_approximation_7,
        pade_approximation_9,
    }, OperationNorm, SVD, Eig};
    use ndarray::{linalg::Dot, *};
    use num_complex::{Complex, ComplexFloat, Complex32 as c32, Complex64 as c64};
    use rand::Rng;
    use ndarray_csv::Array2Reader;
    use csv::ReaderBuilder;

    use super::{expm, MatrixPowers};

    #[test]
    fn test_matrix_powers() {
        let a: Array2<f64> = array![[0., 1.], [1., 0.]];
        let mut mp = MatrixPowers::new(a);
        println!("get3():");
        println!("{:?}", mp.get3());
        println!("get3():");
        println!("{:?}", mp.get3());
        println!("mp?");
        println!("{:?}", mp);
    }

    #[test]
    fn test_expm() {
        fn compute_pade_diff_error(output: &Array2<f64>, expected: &Array2<f64>) -> f64 {
            let mut diff = (output - expected);
            diff.mapv_inplace(|x| x.abs());
            diff.opnorm_one().unwrap()
        }
        let mat: Array2<f64> =
            10. as f64 * array![[0.1, 0.2, 0.3], [0.2, 0.1, 0.5], [0.11, 0.22, 0.32]];
        let expected: Array2<f64> = array![[157.7662816 , 217.94900112, 432.14352751],
        [200.6740715 , 278.69078891, 552.72474352],
        [169.9692465 , 236.2889153 , 469.43670795]];
        println!("expm output:");
        let  out =expm(&mat);
        println!("{:?}", out);
        println!("diff: {:}", compute_pade_diff_error(&out.0, &expected));
    }

    #[test]
    fn test_high_norm() {
        let mut rng = rand::thread_rng();
        let n = 200;
        let samps = 100;
        let mut results = Vec::new();
        let mut avg_entry_error = Vec::new();
        let scale = 0.002;
        for _ in 0..samps {
            let mut m:Array2<c64> = Array2::<c64>::ones((n,n).f());
            m.mapv_inplace(|_| {
                c64::new(rng.gen::<f64>() * 1., rng.gen::<f64>() * 1.)
            });
            m = m.dot(&m.t().map(|x| x.conj()));
            let (mut eigs,mut  vecs) = m.eig().unwrap();
            eigs.mapv_inplace(|_| scale * c64::new(rng.gen::<f64>() , rng.gen::<f64>()));
            let adjoint_vecs = vecs.t().clone().mapv(|x| x.conj());

            let recomputed_m = vecs.dot(&Array2::from_diag(&eigs)).dot(&adjoint_vecs);
            // println!("reconstruction diff: {:}", m_diff.opnorm_one().unwrap() / f64::EPSILON);
            eigs.mapv_inplace(|x| x.exp() );
            let eigen_expm = vecs.dot(&Array2::from_diag(&eigs)).dot(&adjoint_vecs);
            let (expm_comp, deg) = expm(&recomputed_m);
            let diff = &expm_comp - &eigen_expm;
            avg_entry_error.push({
                let tot = diff.map(|x| x.abs()).into_iter().sum::<f64>();
                tot / (n * n) as f64
            });
            results.push(diff.opnorm_one().unwrap());
            // println!("diff norm over epsilon: {:}", diff.opnorm_one().unwrap() / f64::EPSILON);
        }
        let avg: f64 = results.iter().sum::<f64>() / results.len() as f64;
        let avg_entry_diff = avg_entry_error.iter().sum::<f64>() / avg_entry_error.len() as f64;
        let std: f64 = f64::powf(results.iter().map(|x| f64::powi(x - avg, 2)).sum::<f64>() / (results.len() -1)  as f64, 0.5);
        println!("collected {:} samples.", results.len());
        println!("diff norm: {:} +- ({:})", avg, std);
        println!("average entry error over epsilon:{:}", avg_entry_diff / f64::EPSILON);
        println!("avg over epsilon: {:.2}", avg / f64::EPSILON);
        println!("std over epsilon: {:.2}", std / f64::EPSILON);
    }

    #[test]
    fn test_random_matrix() {
        // load data from csv
        let mut input_file = File::open("/Users/matt/Desktop/matrix_input.csv").unwrap();
        let mut output_file = File::open("/Users/matt/Desktop/expm_output.csv").unwrap();
        let mut input_reader = ReaderBuilder::new().has_headers(false).from_reader(input_file);
        let mut output_reader = ReaderBuilder::new().has_headers(false).from_reader(output_file);
        let input: Vec<c64> = Vec::new();
        for res in output_reader.records() {
            res.unwrap().iter().map(|x| {
                let real:Vec<_> = x.split('.').collect();
                println!("real: {:?}", real);
                let mut re: f64 = (real[0].clone()).parse().unwrap();
                println!("re: {:}",re);
                if real[1].contains("*^") {
                    // re += "." + real[1].split("*^").into_iter().take(n)
                }
                let c = c64::from_str(x);
                println!("x: {:}", x);
                println!("c: {:?}", c);
            }).collect::<()>();
            break;
        }
        let input: Array2<c64> = input_reader.deserialize_array2((100,100)).unwrap();
        let expected: Array2<c64> = output_reader.deserialize_array2((100,100)).unwrap();
        let (computed, deg) = expm(&input);
        let diff = &expected - &computed;
        println!("diff norm: {:}", diff.opnorm_one().unwrap());
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
            let theta: c64 =  c64::from_polar(2. * std::f64::consts::PI * rng.gen::<f64>(), 0.);
            let pauli_y: Array2<c64> = array![
                [c64::new(0., 0.), c64::new(0., -1.)],
                [c64::new(0., 1.), c64::new(0., 0.)],
            ];
            let x = c64::new(0., 1.) * theta * pauli_y.clone();
            let actual = c64::cos(theta /2.) * Array2::<c64>::eye(x.nrows()) - c64::sin(theta / 2.) * c64::new(0., 1.) * &pauli_y;
            let (computed, deg) = expm(&(c64::new(0., -0.5) * theta * pauli_y));
            match deg {
                3 => d3 += 1,
                5 => d5 += 1,
                7 => d7 += 1,
                9 => d9 += 1,
                13 => d13 += 1,
                _ => {},
            }
            let diff = (actual - computed).map(|x| x.abs());
            let diff_norm = diff.opnorm_one().unwrap();
            results.push(diff_norm);
        }
        let avg: f64 = results.iter().sum::<f64>() / results.len() as f64;
        let std: f64 = f64::powf(results.iter().map(|x| f64::powi(x - avg, 2)).sum::<f64>() / (results.len() -1)  as f64, 0.5);
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
        let mut mp = MatrixPowers::new(mat);
        pade_approximation_3(&mut mp, &mut output_3);
        pade_approximation_5(&mut mp, &mut output_5);
        pade_approximation_7(&mut mp, &mut output_7);
        pade_approximation_9(&mut mp, &mut output_9);
        pade_approximation_13(&mut mp, &mut output_13);
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