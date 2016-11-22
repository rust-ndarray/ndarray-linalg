//! Define trait for Hermite matrices

use std::fmt::Debug;
use ndarray::prelude::*;
use ndarray::LinalgScalar;
use num_traits::float::Float;
use num_complex::Complex;

use matrix::Matrix;
use error::{LinalgError, NotSquareError};
use qr::ImplQR;
use eig::ImplEig;
use svd::ImplSVD;
use norm::ImplNorm;
use solve::ImplSolve;

/// Methods for square matrices
///
/// This trait defines method for square matrices,
/// but does not assure that the matrix is square.
/// If not square, `NotSquareError` will be thrown.
pub trait SquareMatrix: Matrix {
    fn eig(self) -> Result<(Self::ComplexVector, Self::ComplexMatrix), LinalgError>;
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

impl<A> SquareMatrix for Array<A, (Ix, Ix)>
    where A: ImplEig + ImplQR + ImplNorm + ImplSVD + ImplSolve + LinalgScalar + Float + Debug
{
    fn eig(self) -> Result<(Self::ComplexVector, Self::ComplexMatrix), LinalgError> {
        try!(self.check_square());
        let (n, _) = self.size();
        let (wr, wi, vv) = try!(ImplEig::eig(n, self.into_raw_vec()));
        println!("wi = {:?}", &wi);
        let vr = Array::from_vec(vv).into_shape((n, n)).unwrap();
        let mut v = Array::<Self::Complex, _>::zeros((n, n));
        let mut i = 0;
        while i < n {
            println!("i = {}", &i);
            println!("wi[i] = {:?}", &wi[i]);
            if !wi[i].is_normal() {
                println!("Real eigenvalue");
                for j in 0..n {
                    v[(i, j)] = Complex::new(vr[(i, j)], A::zero());
                }
                i += 1;
            } else {
                println!("Imaginal eigenvalue");
                for j in 0..n {
                    v[(i, j)] = Complex::new(vr[(i, j)], vr[(i + 1, j)]);
                    v[(i + 1, j)] = Complex::new(vr[(i, j)], -vr[(i + 1, j)]);
                }
                i += 2;
            }
        }
        let w = wr.into_iter()
            .zip(wi.into_iter())
            .map(|(r, i)| Complex::new(r, i))
            .collect();
        Ok((w, v.reversed_axes()))
    }
    fn inv(self) -> Result<Self, LinalgError> {
        try!(self.check_square());
        let (n, _) = self.size();
        let is_fortran_align = self.strides()[0] > self.strides()[1];
        let a = try!(ImplSolve::inv(self.layout(), n, self.into_raw_vec()));
        let m = Array::from_vec(a).into_shape((n, n)).unwrap();
        if is_fortran_align {
            Ok(m)
        } else {
            Ok(m.reversed_axes())
        }
    }
    fn trace(&self) -> Result<Self::Scalar, LinalgError> {
        try!(self.check_square());
        let (n, _) = self.size();
        Ok((0..n).fold(A::zero(), |sum, i| sum + self[(i, i)]))
    }
}
