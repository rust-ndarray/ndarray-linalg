
extern crate ndarray;
pub mod lapack_binding;

use ndarray::prelude::*;
use ndarray::LinalgScalar;
use lapack_binding::Eigh;
use std::error;
use std::fmt;

#[derive(Debug)]
pub struct NotSquareError {
    rows: usize,
    cols: usize,
}

#[derive(Debug)]
pub enum LinalgError {
    NotSquare(NotSquareError),
    Lapack(lapack_binding::LapackError),
}

impl fmt::Display for NotSquareError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Not square: rows({}) != cols({})", self.rows, self.cols)
    }
}

impl fmt::Display for LinalgError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            LinalgError::NotSquare(ref err) => err.fmt(f),
            LinalgError::Lapack(ref err) => err.fmt(f),
        }
    }
}

impl error::Error for NotSquareError {
    fn description(&self) -> &str {
        "Matrix is not square"
    }
}

impl error::Error for LinalgError {
    fn description(&self) -> &str {
        match *self {
            LinalgError::NotSquare(ref err) => err.description(),
            LinalgError::Lapack(ref err) => err.description(),
        }
    }
}

impl From<NotSquareError> for LinalgError {
    fn from(err: NotSquareError) -> LinalgError {
        LinalgError::NotSquare(err)
    }
}

impl From<lapack_binding::LapackError> for LinalgError {
    fn from(err: lapack_binding::LapackError) -> LinalgError {
        LinalgError::Lapack(err)
    }
}

pub trait Matrix: Sized {
    type Vector;
    /// number of rows and cols
    fn size(&self) -> (usize, usize);
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

impl<A> Matrix for Array<A, (Ix, Ix)> {
    type Vector = Array<A, Ix>;
    fn size(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }
}

impl<A> SquareMatrix for Array<A, (Ix, Ix)>
    where A: Eigh + LinalgScalar
{
    fn eigh(self) -> Result<(Self::Vector, Self), LinalgError> {
        try!(self.check_square());
        let (rows, cols) = self.size();
        let mut a = self.into_raw_vec();
        let w = try!(Eigh::syev(rows as i32, &mut a));
        let ea = Array::from_vec(w);
        let va = Array::from_vec(a).into_shape((rows, cols)).unwrap().reversed_axes();
        Ok((ea, va))
    }
}
