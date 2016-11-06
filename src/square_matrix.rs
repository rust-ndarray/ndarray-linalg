
use ndarray::prelude::*;
use ndarray::LinalgScalar;
use num_traits::float::Float;

use matrix::Matrix;
use error::{LinalgError, NotSquareError};
use eigh::ImplEigh;
use svd::ImplSVD;
use norm::ImplNorm;
use solve::ImplSolve;

pub trait SquareMatrix: Matrix {
    // fn qr(self) -> (Self, Self);
    // fn lu(self) -> (Self, Self);
    // fn eig(self) -> (Self::Vector, Self);
    /// eigenvalue decomposition for Hermite matrix
    fn eigh(self) -> Result<(Self::Vector, Self), LinalgError>;
    /// inverse of matrix
    fn inv(self) -> Result<Self, LinalgError>;
    fn trace(&self) -> Result<Self::Scalar, LinalgError>;
    fn ssqrt(self) -> Result<Self, LinalgError>;
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

impl<A: ImplSVD + ImplNorm + ImplSolve +  ImplEigh + LinalgScalar + Float> SquareMatrix for Array<A, (Ix, Ix)> {
    fn eigh(self) -> Result<(Self::Vector, Self), LinalgError> {
        try!(self.check_square());
        let (rows, cols) = self.size();
        let (w, a) = try!(ImplEigh::eigh(rows, self.into_raw_vec()));
        let ea = Array::from_vec(w);
        let va = Array::from_vec(a).into_shape((rows, cols)).unwrap().reversed_axes();
        Ok((ea, va))
    }
    fn inv(self) -> Result<Self, LinalgError> {
        try!(self.check_square());
        let (n, _) = self.size();
        let is_fortran_align = self.strides()[0] > self.strides()[1];
        let a = try!(ImplSolve::inv(n, self.into_raw_vec()));
        let m = Array::from_vec(a).into_shape((n, n)).unwrap();
        if is_fortran_align {
            Ok(m)
        } else {
            Ok(m.reversed_axes())
        }
    }
    fn ssqrt(self) -> Result<Self, LinalgError> {
        let (n, _) = self.size();
        let (e, v) = try!(self.eigh());
        let mut res = Array::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                res[(i, j)] = e[i].sqrt() * v[(j, i)];
            }
        }
        Ok(v.dot(&res))
    }
    fn trace(&self) -> Result<Self::Scalar, LinalgError> {
        try!(self.check_square());
        let (n, _) = self.size();
        Ok((0..n).fold(A::zero(), |sum, i| sum + self[(i, i)]))
    }
}
