
use ndarray::prelude::*;

use scalar::LapackScalar;

pub trait Matrix: Sized {
    type Scalar;
    type Vector;
    /// number of rows and cols
    fn size(&self) -> (usize, usize);
    fn norm_1(&self) -> Self::Scalar;
    fn norm_i(&self) -> Self::Scalar;
    fn norm_f(&self) -> Self::Scalar;
    // fn svd(self) -> (Self, Self::Vector, Self);
}

impl<A: LapackScalar> Matrix for Array<A, (Ix, Ix)> {
    type Scalar = A;
    type Vector = Array<A, Ix>;
    fn size(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }
    fn norm_1(&self) -> Self::Scalar {
        let (m, n) = self.size();
        let strides = self.strides();
        if strides[0] > strides[1] {
            LapackScalar::norm_i(n, m, self.clone().into_raw_vec())
        } else {
            LapackScalar::norm_1(m, n, self.clone().into_raw_vec())
        }
    }
    fn norm_i(&self) -> Self::Scalar {
        let (m, n) = self.size();
        let strides = self.strides();
        if strides[0] > strides[1] {
            LapackScalar::norm_1(n, m, self.clone().into_raw_vec())
        } else {
            LapackScalar::norm_i(m, n, self.clone().into_raw_vec())
        }
    }
    fn norm_f(&self) -> Self::Scalar {
        let (m, n) = self.size();
        LapackScalar::norm_f(m, n, self.clone().into_raw_vec())
    }
}
