
extern crate ndarray;
extern crate num_traits;

pub mod lapack_binding;
pub mod error;

use ndarray::prelude::*;
use ndarray::LinalgScalar;
use num_traits::float::Float;
use lapack_binding::LapackScalar;
use error::{LinalgError, NotSquareError};

pub trait Vector {
    type Scalar;
    fn norm(&self) -> Self::Scalar;
}

impl<A: Float + LinalgScalar> Vector for Array<A, Ix> {
    type Scalar = A;
    fn norm(&self) -> Self::Scalar {
        self.dot(&self).sqrt()
    }
}

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

pub trait SquareMatrix: Matrix {
    // fn qr(self) -> (Self, Self);
    // fn lu(self) -> (Self, Self);
    // fn eig(self) -> (Self::Vector, Self);
    /// eigenvalue decomposition for Hermite matrix
    fn eigh(self) -> Result<(Self::Vector, Self), LinalgError>;
    fn check_square(&self) -> Result<(), NotSquareError> {
        let (rows, cols) = self.size();
        if rows == cols {
            Ok(())
        } else {
            Err(NotSquareError {
                rows: rows,
                cols: cols,
            })
        }
    }
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

impl<A: LapackScalar> SquareMatrix for Array<A, (Ix, Ix)> {
    fn eigh(self) -> Result<(Self::Vector, Self), LinalgError> {
        try!(self.check_square());
        let (rows, cols) = self.size();
        let (w, a) = try!(LapackScalar::eigh(rows, self.into_raw_vec()));
        let ea = Array::from_vec(w);
        let va = Array::from_vec(a).into_shape((rows, cols)).unwrap().reversed_axes();
        Ok((ea, va))
    }
}
