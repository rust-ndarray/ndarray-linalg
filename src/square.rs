//! Define trait for Hermite matrices

use ndarray::{Ix2, Array, LinalgScalar};
use std::fmt::Debug;
use num_traits::float::Float;
use lapack::c::Layout;

use matrix::Matrix;
use error::{LinalgError, NotSquareError};
use qr::ImplQR;
use svd::ImplSVD;
use norm::ImplNorm;
use solve::ImplSolve;

/// Methods for square matrices
///
/// This trait defines method for square matrices,
/// but does not assure that the matrix is square.
/// If not square, `NotSquareError` will be thrown.
pub trait SquareMatrix: Matrix {
    // fn eig(self) -> (Self::Vector, Self);
    /// inverse matrix
    fn inv(self) -> Result<Self, LinalgError>;
    /// trace of matrix
    fn trace(&self) -> Result<Self::Scalar, LinalgError>;
    /// test matrix is square
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

impl<A> SquareMatrix for Array<A, Ix2>
    where A: ImplQR + ImplNorm + ImplSVD + ImplSolve + LinalgScalar + Float + Debug
{
    fn inv(self) -> Result<Self, LinalgError> {
        try!(self.check_square());
        let (n, _) = self.size();
        let layout = self.layout()?;
        let a = ImplSolve::inv(layout, n, self.into_raw_vec())?;
        let m = Array::from_vec(a).into_shape((n, n)).unwrap();
        match layout {
            Layout::RowMajor => Ok(m),
            Layout::ColumnMajor => Ok(m.reversed_axes()),
        }
    }
    fn trace(&self) -> Result<Self::Scalar, LinalgError> {
        try!(self.check_square());
        let (n, _) = self.size();
        Ok((0..n).fold(A::zero(), |sum, i| sum + self[(i, i)]))
    }
}
