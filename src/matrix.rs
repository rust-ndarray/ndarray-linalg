//! Define trait for general matrix

use std::cmp::min;
use ndarray::prelude::*;
use ndarray::LinalgScalar;

use error::LapackError;
use qr::ImplQR;
use svd::ImplSVD;
use norm::ImplNorm;

/// Methods for general matrices
pub trait Matrix: Sized {
    type Scalar;
    type Vector;
    /// number of (rows, columns)
    fn size(&self) -> (usize, usize);
    /// Operator norm for L-1 norm
    fn norm_1(&self) -> Self::Scalar;
    /// Operator norm for L-inf norm
    fn norm_i(&self) -> Self::Scalar;
    /// Frobenius norm
    fn norm_f(&self) -> Self::Scalar;
    /// singular-value decomposition (SVD)
    fn svd(self) -> Result<(Self, Self::Vector, Self), LapackError>;
    fn qr(self) -> Result<(Self, Self), LapackError>;
}

impl<A> Matrix for Array<A, (Ix, Ix)>
    where A: ImplQR + ImplSVD + ImplNorm + LinalgScalar
{
    type Scalar = A;
    type Vector = Array<A, Ix>;
    fn size(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }
    fn norm_1(&self) -> Self::Scalar {
        let (m, n) = self.size();
        let strides = self.strides();
        if strides[0] > strides[1] {
            ImplNorm::norm_i(n, m, self.clone().into_raw_vec())
        } else {
            ImplNorm::norm_1(m, n, self.clone().into_raw_vec())
        }
    }
    fn norm_i(&self) -> Self::Scalar {
        let (m, n) = self.size();
        let strides = self.strides();
        if strides[0] > strides[1] {
            ImplNorm::norm_1(n, m, self.clone().into_raw_vec())
        } else {
            ImplNorm::norm_i(m, n, self.clone().into_raw_vec())
        }
    }
    fn norm_f(&self) -> Self::Scalar {
        let (m, n) = self.size();
        ImplNorm::norm_f(m, n, self.clone().into_raw_vec())
    }
    fn svd(self) -> Result<(Self, Self::Vector, Self), LapackError> {
        let strides = self.strides();
        let (m, n) = if strides[0] > strides[1] {
            self.size()
        } else {
            let (n, m) = self.size();
            (m, n)
        };
        let (u, s, vt) = try!(ImplSVD::svd(m, n, self.clone().into_raw_vec()));
        let sv = Array::from_vec(s);
        if strides[0] > strides[1] {
            let ua = Array::from_vec(u).into_shape((n, n)).unwrap();
            let va = Array::from_vec(vt).into_shape((m, m)).unwrap();
            Ok((va, sv, ua))
        } else {
            let ua = Array::from_vec(u).into_shape((n, n)).unwrap().reversed_axes();
            let va = Array::from_vec(vt).into_shape((m, m)).unwrap().reversed_axes();
            Ok((ua, sv, va))
        }
    }
    fn qr(self) -> Result<(Self, Self), LapackError> {
        let (n, m) = self.size();
        let strides = self.strides();
        let k = min(n, m);
        println!("n = {:?}", n);
        println!("m = {:?}", m);
        println!("strides = {:?}", strides);
        let (mut q, mut r) = if strides[0] < strides[1] {
            try!(ImplQR::qr(m, n, self.clone().into_raw_vec()))
        } else {
            try!(ImplQR::lq(n, m, self.clone().into_raw_vec()))
        };
        println!("q.len = {:?}", q.len());
        println!("r.len = {:?}", r.len());
        q.truncate(n * k);
        r.truncate(m * k);
        let (qa, mut ra) = if strides[0] < strides[1] {
            (Array::from_vec(q).into_shape((k, n)).unwrap().reversed_axes(),
             Array::from_vec(r).into_shape((m, k)).unwrap().reversed_axes())
        } else {
            (Array::from_vec(q).into_shape((n, k)).unwrap(),
             Array::from_vec(r).into_shape((k, m)).unwrap())
        };
        for ((i, j), val) in ra.indexed_iter_mut() {
            if i > j {
                *val = A::zero();
            }
        }
        Ok((qa, ra))
    }
}
