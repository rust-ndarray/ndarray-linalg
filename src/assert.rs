//! Assertions for value and array

use ndarray::{Array, Dimension, IntoDimension};
use float_cmp::ApproxEqRatio;
use num_complex::Complex;

/// test two values are close in relative tolerance sense
pub trait AssertClose: Sized + Copy {
    type Tol;
    fn assert_close(self, truth: Self, rtol: Self::Tol);
}

macro_rules! impl_AssertClose {
    ($scalar:ty) => {
impl AssertClose for $scalar {
    type Tol = $scalar;
    fn assert_close(self, truth: Self, rtol: Self::Tol) {
        if !self.approx_eq_ratio(&truth, rtol) {
            panic!("Not close: val={}, truth={}, rtol={}", self, truth, rtol);
        }
    }
}
impl AssertClose for Complex<$scalar> {
    type Tol = $scalar;
    fn assert_close(self, truth: Self, rtol: Self::Tol) {
        if !(self.re.approx_eq_ratio(&truth.re, rtol) && self.im.approx_eq_ratio(&truth.im, rtol)) {
            panic!("Not close: val={}, truth={}, rtol={}", self, truth, rtol);
        }
    }
}
}} // impl_AssertClose
impl_AssertClose!(f64);
impl_AssertClose!(f32);

/// test two arrays are close
pub trait AssertAllClose {
    type Tol;
    /// test two arrays are close in L2-norm with relative tolerance
    fn assert_allclose_l2(&self, truth: &Self, rtol: Self::Tol);
    /// test two arrays are close in inf-norm with absolute tolerance
    fn assert_allclose_inf(&self, truth: &Self, atol: Self::Tol);
}

macro_rules! impl_AssertAllClose {
    ($scalar:ty, $float:ty, $abs:ident) => {
impl AssertAllClose for [$scalar]{
    type Tol = $float;
    fn assert_allclose_inf(&self, truth: &Self, atol: Self::Tol) {
        for (x, y) in self.iter().zip(truth.iter()) {
            let tol = (x - y).$abs();
            if tol > atol {
                panic!("Not close in inf-norm (atol={}): \ntest = \n{:?}\nTruth = \n{:?}",
                       atol, self, truth);
            }
        }
    }
    fn assert_allclose_l2(&self, truth: &Self, rtol: Self::Tol) {
        let nrm: Self::Tol = truth.iter().map(|x| x.$abs().powi(2)).sum();
        let dev: Self::Tol = self.iter().zip(truth.iter()).map(|(x, y)| (x-y).$abs().powi(2)).sum();
        if dev / nrm > rtol.powi(2) {
            panic!("Not close in L2-norm (rtol={}): \ntest = \n{:?}\nTruth = \n{:?}",
                   rtol, self, truth);
        }
    }
}

impl AssertAllClose for Vec<$scalar> {
    type Tol = $float;
    fn assert_allclose_inf(&self, truth: &Self, atol: Self::Tol) {
        self.as_slice().assert_allclose_inf(&truth, atol);
    }
    fn assert_allclose_l2(&self, truth: &Self, rtol: Self::Tol) {
        self.as_slice().assert_allclose_l2(&truth, rtol);
    }
}

impl<D: Dimension> AssertAllClose for Array<$scalar, D> {
    type Tol = $float;
    fn assert_allclose_inf(&self, truth: &Self, atol: Self::Tol) {
        if self.shape() != truth.shape() {
            panic!("Shape missmatch: self={:?}, truth={:?}", self.shape(), truth.shape());
        }
        for (idx, val) in self.indexed_iter() {
            let t = truth[idx.into_dimension()];
            let tol = (*val - t).$abs();
            if tol > atol {
                panic!("Not close in inf-norm (atol={}): \ntest = \n{:?}\nTruth = \n{:?}",
                       atol, self, truth);
            }
        }
    }
    fn assert_allclose_l2(&self, truth: &Self, rtol: Self::Tol) {
        if self.shape() != truth.shape() {
            panic!("Shape missmatch: self={:?}, truth={:?}", self.shape(), truth.shape());
        }
        let nrm: Self::Tol = truth.iter().map(|x| x.$abs().powi(2)).sum();
        let dev: Self::Tol = self.indexed_iter().map(|(idx, val)| (truth[idx.into_dimension()] - val).$abs().powi(2)).sum();
        if dev / nrm > rtol.powi(2) {
            panic!("Not close in L2-norm (rtol={}): \ntest = \n{:?}\nTruth = \n{:?}",
                   rtol, self, truth);
        }
    }
}
}} // impl_AssertAllClose

impl_AssertAllClose!(f64, f64, abs);
impl_AssertAllClose!(f32, f32, abs);
impl_AssertAllClose!(Complex<f64>, f64, norm);
impl_AssertAllClose!(Complex<f32>, f32, norm);
