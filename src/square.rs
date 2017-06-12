//! Define trait for Hermite matrices

use ndarray::{Ix2, Array, RcArray, ArrayBase, Data};

use super::matrix::{Matrix, MFloat};
use super::error::{LinalgError, NotSquareError};

/// Methods for square matrices
///
/// This trait defines method for square matrices,
/// but does not assure that the matrix is square.
/// If not square, `NotSquareError` will be thrown.
pub trait SquareMatrix: Matrix {
    /// trace of matrix
    fn trace(&self) -> Result<Self::Scalar, LinalgError>;
    #[doc(hidden)]
    fn check_square(&self) -> Result<(), NotSquareError> {
        let (rows, cols) = self.size();
        if rows == cols {
            Ok(())
        } else {
            Err(NotSquareError {
                rows: rows as i32,
                cols: cols as i32,
            })
        }
    }
    /// test matrix is square and return its size
    fn square_size(&self) -> Result<usize, NotSquareError> {
        self.check_square()?;
        let (n, _) = self.size();
        Ok(n)
    }
}

fn trace<A: MFloat, S>(a: &ArrayBase<S, Ix2>) -> A
    where S: Data<Elem = A>
{
    let n = a.rows();
    (0..n).fold(A::zero(), |sum, i| sum + a[(i, i)])
}

impl<A: MFloat> SquareMatrix for Array<A, Ix2> {
    fn trace(&self) -> Result<Self::Scalar, LinalgError> {
        self.check_square()?;
        Ok(trace(self))
    }
}

impl<A: MFloat> SquareMatrix for RcArray<A, Ix2> {
    fn trace(&self) -> Result<Self::Scalar, LinalgError> {
        self.check_square()?;
        Ok(trace(self))
    }
}
