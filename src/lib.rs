
extern crate ndarray;

pub mod lapack_binding;
pub mod error;

use ndarray::prelude::*;
use lapack_binding::LapackScalar;
use error::{LinalgError, NotSquareError};

pub trait Matrix: Sized {
    type Scalar;
    type Vector;
    /// number of rows and cols
    fn size(&self) -> (usize, usize);
    // fn svd(self) -> (Self, Self::Vector, Self);
    // fn norm1(&self) -> Scalar;
    // fn norm2(&self) -> Scalar;
    // fn normI(&self) -> Scalar;
    // fn normF(&self) -> Scalar;
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

impl<A> Matrix for Array<A, (Ix, Ix)> {
    type Scalar = A;
    type Vector = Array<A, Ix>;
    fn size(&self) -> (usize, usize) {
        (self.rows(), self.cols())
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
