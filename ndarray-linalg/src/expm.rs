use std::ops::MulAssign;

use crate::{normest1::normest, types, Inverse, OperationNorm};
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

struct OneNormOneCalc {}

/// Computes operator one norms either exactly or estimates them using Higham & Tisseur (ref)
pub enum OpNormOne<'a> {
    Exact {
        a_matrix: &'a Array2<f64>,
    },
    Estimate {
        a_matrix: &'a Array2<f64>,
        t: usize,
        itmax: usize,
    },
    EstimateSequence {
        matrices: Vec<&'a Array2<f64>>,
    },
    EstimatePower {
        a_matrix: &'a Array2<f64>,
        p: usize,
    },
}

impl<'a> OpNormOne<'a> {
    fn compute(&self) -> f64 {
        match self {
            Self::Exact { a_matrix } => a_matrix.opnorm_one().unwrap(),
            Self::Estimate { a_matrix, t, itmax } => normest(a_matrix, *t, *itmax as u32),
            Self::EstimateSequence { matrices } => 1.,
            _ => 1.,
        }
    }
}

fn power_abs_norm(matrix: &Array2<f64>, p: usize) -> f64 {
    let mut v = Array1::<f64>::ones((matrix.ncols()).f());
    let abs_matrix = matrix.t().map(|x| x.abs());
    for _ in 0..p {
        v.assign(&abs_matrix.dot(&v));
    }
    v.into_iter()
        .reduce(|x, y| if x > y { x } else { y })
        .unwrap()
}

// helper function used in Al-M & H. Name is unchanged for future reference.
fn ell(a_matrix: &Array2<f64>, m: u64) -> i32 {
    if a_matrix.is_square() == false {
        panic!("subroutine ell expected a square matrix.");
    }
    let p = 2 * m + 1;
    let a_one_norm = a_matrix.opnorm_one().unwrap();

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

fn pade_approximation_3(mp: &mut MatrixPowers) -> Array2<f64> {
    let mut evens = PADE_COEFFS_3[0] * Array::eye(mp.get(1).nrows());
    evens.scaled_add(PADE_COEFFS_3[2], mp.get(2));
    let mut odds = PADE_COEFFS_3[1] * Array::eye(mp.get(1).nrows());
    odds.scaled_add(PADE_COEFFS_3[3], mp.get(2));
    odds = odds.dot(mp.get(1));
    let inverted = (-1. * odds + evens).inv().unwrap();
    inverted.dot(&(odds + evens))
}

fn pade_approximation_5(mp: &mut MatrixPowers) -> Array2<f64> {
    let mut evens: Array2<f64> = PADE_COEFFS_9[0] * Array::eye(mp.get(1).nrows());
    for m in 1..=2 {
        evens.scaled_add(PADE_COEFFS_9[2 * m], mp.get(2 * m));
    }
    let mut odds: Array2<f64> = PADE_COEFFS_9[1] * Array::eye(mp.get(1).nrows());
    for m in 1..=2 {
        odds.scaled_add(PADE_COEFFS_9[2 * m + 1], mp.get(2*m));
    }
    odds = odds.dot(mp.get(1));
    let inverted = (-1. * odds + evens).inv().unwrap();
    inverted.dot(&(odds + evens))
}

fn pade_approximation_7(mp: &mut MatrixPowers) -> Array2<f64> {
    let mut evens: Array2<f64> = PADE_COEFFS_9[0] * Array::eye(mp.get(1).nrows());
    for m in 1..=3 {
        evens.scaled_add(PADE_COEFFS_9[2 * m], mp.get(2 * m));
    }
    let mut odds: Array2<f64> = PADE_COEFFS_9[1] * Array::eye(mp.get(1).nrows());
    for m in 1..=3 {
        odds.scaled_add(PADE_COEFFS_9[2 * m + 1], mp.get(2*m));
    }
    odds = odds.dot(mp.get(1));
    let inverted = (-1. * odds + evens).inv().unwrap();
    inverted.dot(&(odds + evens))
}
fn pade_approximation_9(mp: &mut MatrixPowers) -> Array2<f64> {
    let mut evens: Array2<f64> = PADE_COEFFS_9[0] * Array::eye(mp.get(1).nrows());
    for m in 1..=4 {
        evens.scaled_add(PADE_COEFFS_9[2 * m], mp.get(2 * m));
    }
    let mut odds: Array2<f64> = PADE_COEFFS_9[1] * Array::eye(mp.get(1).nrows());
    for m in 1..=4 {
        odds.scaled_add(PADE_COEFFS_9[2 * m + 1], mp.get(2*m));
    }
    odds = odds.dot(mp.get(1));
    let inverted = (-1. * odds + evens).inv().unwrap();
    inverted.dot(&(odds + evens))
}

fn pade_approximation_13(mp: &mut MatrixPowers) -> Array2<f64> {
    let mut evens_1 = PADE_COEFFS_13[0] * Array::eye(mp.get(1).nrows());
    for m in 1..=3 {
        evens_1.scaled_add(PADE_COEFFS_13[2 * m], mp.get(2 * m));
    }
    let mut evens_2 = PADE_COEFFS_13[8] * mp.get(2).clone();
    evens_2.scaled_add(PADE_COEFFS_13[10], mp.get(4));
    evens_2.scaled_add(PADE_COEFFS_13[12], mp.get(6));
    let evens = evens_2.dot(mp.get(6)) + &evens_1;
    let mut odds_1 = PADE_COEFFS_13[1] * Array::eye(mp.get(1).nrows());
    for m in 1..=3 {
        odds_1.scaled_add(PADE_COEFFS_13[2 * m + 1], mp.get(2 * m));
    }
    let mut odds_2 = PADE_COEFFS_13[9] * mp.get(2).clone();
    odds_2.scaled_add(PADE_COEFFS_13[11], mp.get(4));
    odds_2.scaled_add(PADE_COEFFS_13[13], mp.get(6));
    odds_2 = odds_2.dot(mp.get(6));
    let odds = (odds_1 + odds_2).dot(mp.get(1));
    let inverted = (-1. * odds.clone() + evens.clone()).inv().unwrap();
    inverted.dot(&(odds + evens))
}

/// Helper struct to ensure that the power of the input matrix is only computed once.
#[derive(Debug)]
struct MatrixPowers<'a> {
    pub input_1 : Option<&'a Array2<f64>>,
    pub input_2 : Option<Array2<f64>>,
    pub input_3 : Option<Array2<f64>>,
    pub input_4 : Option<Array2<f64>>,
    pub input_5 : Option<Array2<f64>>,
    pub input_6 : Option<Array2<f64>>,
    pub input_7 : Option<Array2<f64>>,
    pub input_8 : Option<Array2<f64>>,
    pub input_9 : Option<Array2<f64>>,
    pub input_10 : Option<Array2<f64>>,
    pub input_11 : Option<Array2<f64>>,
    pub input_12 : Option<Array2<f64>>,
    pub input_13 : Option<Array2<f64>>,
    pub num_matprods: usize,
}
impl<'a> MatrixPowers<'a> {
    pub fn new() -> Self {
        MatrixPowers { input_1: None, input_2: None, input_3:None, input_4: None, input_5: None, input_6: None, input_7: None, input_8: None, input_9: None, input_10: None, input_11: None, input_12: None, input_13: None, num_matprods: 0 }
    }

    fn compute2(&mut self) {
        if let Some(input_1) = self.input_1.clone() {
            self.input_2 = Some(input_1.dot(input_1));
            self.num_matprods += 1;
        }
    }
    fn compute3(&mut self) {
        match &self.input_2 {
            Some(i2) => {
                self.input_3 = Some(self.input_1.as_ref().unwrap().dot(i2));
                self.num_matprods += 1;
            },
            None => {
                // after calling self.compute2() then self.input_2 will match to Some(_)
                // so just recurse.
                self.compute2();
                self.compute3();
            },
        }
    }
    fn compute4(&mut self) {
        match &self.input_2 {
            Some(i2) => {
                self.input_4 = Some(self.input_2.as_ref().unwrap().dot(i2));
                self.num_matprods += 1;
            },
            None => {
                self.compute2();
                self.compute4();
            }
        }
    }
    fn compute5(&mut self) {
        match &self.input_3 {
            Some(i3) => {
                match &self.input_2 {
                    Some(i2) => {
                        self.input_5 = Some(i2.dot(i3));
                        self.num_matprods += 1;
                    },
                    None => {
                        self.compute2();
                        self.compute5();
                    }
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
            },
            None => {
                match &self.input_3 {
                    Some(i3) => {
                        self.input_6 = Some(i3.dot(i3));
                        self.num_matprods += 1;
                    },
                    None => {
                        // We do not have 4 or 3 computed yet, so we will either have 2 or have to compute it.
                        // in that case easiest way is to first compute 4 and then revisit.
                        self.compute4();
                        self.compute6();
                    },
                }
            }
        };
    }

    pub fn get(&mut self, m: usize) -> &Array2<f64> {
        match m {
            1 => self.get1(),
            2 => self.get2(),
            3 => self.get3(),
            4 => self.get4(),
            5 => self.get5(),
            6 => self.get6(),
            // 7 => self.get7(),
            // 8 => self.get8(),
            // 9 => self.get9(),
            // 10 => self.get10(),
            // 11 => self.get11(),
            // 12 => self.get12(),
            // 13 => self.get13(),
            _ => {
                println!("This power of input matrix is not implemented. Returning input matrix.");
                self.input_1.as_ref().unwrap()
            }
        }
    }
    fn get1(&mut self) -> &Array2<f64> {
        self.input_1.as_ref().unwrap()
    }
    fn get2(&mut self) -> &Array2<f64> {
        match &self.input_2 {
            Some(mat) => self.input_2.as_ref().unwrap(),
            None => {
                self.compute2();
                self.input_2.as_ref().unwrap()
            }
        }
    }
    fn get3(&mut self) -> &Array2<f64> {
        match &self.input_3 {
            Some(mat) => self.input_3.as_ref().unwrap(),
            None => {
                self.compute3();
                &self.input_3.as_ref().unwrap()
            }
        }
    }
    fn get4(&mut self) -> &Array2<f64> {
        match &self.input_4 {
            Some(mat) => &self.input_4.as_ref().unwrap(),
            None => {
                self.compute4();
                &self.input_4.as_ref().unwrap()
            }
        }
    }
    fn get5(&mut self) -> &Array2<f64> {
        match &self.input_5 {
            Some(mat) => &self.input_5.as_ref().unwrap(),
            None => {
                self.compute5();
                &self.input_5.as_ref().unwrap()
            }
        }
    }
    fn get6(&mut self) -> &Array2<f64> {
        match &self.input_6 {
            Some(mat) => &self.input_6.as_ref().unwrap(),
            None => {
                self.compute6();
                &self.input_6.as_ref().unwrap()
            }
        }
    }
}

fn pade_approximation(mp: &MatrixPowers, output: &mut Array2<f64>, degree: &usize) {
    match *degree {
        3 => {
            pade_approximation_3(mp);
        }
        5 => {
            pade_approximation_5(mp.get(1), mp.get(2), mp.get(4), output);
        }
        7 => {
            let input_2 = mp.dot(mp);
            let input_4 = input_2.dot(&input_2);
            let input_6 = input_4.dot(&input_2);
            pade_approximation_7(mp, &input_2, &input_4, &input_6, output);
        }
        9 => {
            let input_2 = mp.dot(mp);
            let input_4 = input_2.dot(&input_2);
            let input_6 = input_4.dot(&input_2);
            let input_8 = input_4.dot(&input_4);
            pade_approximation_9(mp, &input_2, &input_4, &input_6, &input_8, output);
        }
        13 => {
            let input_2 = mp.dot(mp);
            let input_4 = input_2.dot(&input_2);
            let input_6 = input_4.dot(&input_2);
            pade_approximation_13(mp, &input_2, &input_4, &input_6, output);
        }
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
pub fn expm(A: Array2<f64>) -> Array2<f64> {
    let mut output = Array2::<f64>::zeros((A.nrows(), A.ncols()).f());
    let d6 = f64::powf(power_abs_norm(&A, 6), 1. / 6.);
    let d4 = f64::powf(power_abs_norm(&A, 4), 1. / 4.);
    let eta_1 = f64::max(d4, d6);
    if eta_1 < THETA_3 && ell(&A, 3) == 0 {
        pade_approximation(&A, &mut output, &3);
        return output;
    }

    arr2(&[[2.]])
}

mod tests {
    use crate::expm::{
        pade_approximation_13, pade_approximation_3, pade_approximation_5, pade_approximation_7,
        pade_approximation_9,
    };
    use ndarray::{linalg::Dot, *};

    use super::{expm, MatrixPowers};

    #[test]
    fn test_matrix_powers() {
        let mut mp = MatrixPowers::new();
        mp.input_1 = Some(array![[0., 1.], [1., 0.]]);
        println!("get3():");
        println!("{:?}",mp.get3());
        println!("get3():");
        println!("{:?}",mp.get3());
        println!("mp?");
        println!("{:?}", mp);
    }
    
    #[test]
    fn test_expm() {
        fn compute_pade_diff_error(output: &Array2<f64>, expected: &Array2<f64>) -> f64 {
            let mut tot = 0.0;
            for row_ix in 0..output.nrows() {
                for col_ix in 0..output.ncols() {
                    tot += (output[[row_ix, col_ix]] - expected[[row_ix, col_ix]]).abs();
                }
            }
            tot
        }
        let mat: Array2<f64> =
            0.01 as f64 * array![[0.1, 0.2, 0.3], [0.2, 0.1, 0.5], [0.11, 0.22, 0.32]];
        let expected: Array2<f64> = array![
            [1.001, 0.00200531, 0.00301133],
            [0.00200476, 1.00101, 0.00501353],
            [0.00110452, 0.00220573, 1.00321]
        ];
        println!("expm output:");
        println!("{:?}", expm(mat.clone()));
        println!("diff: {:}", compute_pade_diff_error(&expm(mat), &expected));
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
        // pade_approximation_3(&mat, &mut output_3);
        // pade_approximation_5(&mat, &mut output_5);
        // pade_approximation_7(&mat, &mut output_7);
        // pade_approximation_9(&mat, &mut output_9);
        // pade_approximation_13(&mat, &mut output_13);
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
