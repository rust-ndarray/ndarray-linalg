//! Define trait for Hermite matrices

use ndarray::{Ix2, Array, RcArray, ArrayBase, Data};
use lapack::c::Layout;

use matrix::{Matrix, MFloat};
use error::{LinalgError, NotSquareError};
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
    #[doc(hidden)]
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
    fn inv(self) -> Result<Self, LinalgError> {
        self.check_square()?;
        let (n, _) = self.size();
        let layout = self.layout()?;
        let (ipiv, a) = ImplSolve::lu(layout, n, n, self.into_raw_vec())?;
        let a = ImplSolve::inv(layout, n, a, &ipiv)?;
        let m = Array::from_vec(a).into_shape((n, n)).unwrap();
        match layout {
            Layout::RowMajor => Ok(m),
            Layout::ColumnMajor => Ok(m.reversed_axes()),
        }
    }
    fn trace(&self) -> Result<Self::Scalar, LinalgError> {
        self.check_square()?;
        Ok(trace(self))
    }
}

impl<A: MFloat> SquareMatrix for RcArray<A, Ix2> {
    fn inv(self) -> Result<Self, LinalgError> {
        // XXX unnecessary clone (should use into_owned())
        let i = self.to_owned().inv()?;
        Ok(i.into_shared())
    }
    fn trace(&self) -> Result<Self::Scalar, LinalgError> {
        self.check_square()?;
        Ok(trace(self))
    }
}
