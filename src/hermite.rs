//! Define trait for Hermite matrices

use ndarray::prelude::*;
use ndarray::LinalgScalar;
use num_traits::float::Float;

use matrix::Matrix;
use square::SquareMatrix;
use error::LinalgError;
use eigh::ImplEigh;
use svd::ImplSVD;
use norm::ImplNorm;
use solve::ImplSolve;

/// Methods for Hermite matrix
pub trait HermiteMatrix: SquareMatrix + Matrix {
    /// eigenvalue decomposition
    fn eigh(self) -> Result<(Self::Vector, Self), LinalgError>;
    /// symmetric square root of Hermite matrix
    fn ssqrt(self) -> Result<Self, LinalgError>;
}

impl<A> HermiteMatrix for Array<A, (Ix, Ix)>
    where A: ImplSVD + ImplNorm + ImplSolve + ImplEigh + LinalgScalar + Float
{
    fn eigh(self) -> Result<(Self::Vector, Self), LinalgError> {
        try!(self.check_square());
        let (rows, cols) = self.size();
        let (w, a) = try!(ImplEigh::eigh(rows, self.into_raw_vec()));
        let ea = Array::from_vec(w);
        let va = Array::from_vec(a).into_shape((rows, cols)).unwrap().reversed_axes();
        Ok((ea, va))
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
}
