//! Define trait for general matrix

use std::cmp::min;
use ndarray::prelude::*;
use lapack::c::Layout;

use error::{LinalgError, StrideError};
use qr::ImplQR;
use svd::ImplSVD;
use norm::ImplNorm;
use solve::ImplSolve;

pub trait MFloat: ImplQR + ImplSVD + ImplNorm + ImplSolve + NdFloat {}
impl MFloat for f32 {}
impl MFloat for f64 {}

/// Methods for general matrices
pub trait Matrix: Sized {
    type Scalar;
    type Vector;
    type Permutator;
    /// number of (rows, columns)
    fn size(&self) -> (usize, usize);
    /// Layout (C/Fortran) of matrix
    fn layout(&self) -> Result<Layout, StrideError>;
    /// Operator norm for L-1 norm
    fn norm_1(&self) -> Self::Scalar;
    /// Operator norm for L-inf norm
    fn norm_i(&self) -> Self::Scalar;
    /// Frobenius norm
    fn norm_f(&self) -> Self::Scalar;
    /// singular-value decomposition (SVD)
    fn svd(self) -> Result<(Self, Self::Vector, Self), LinalgError>;
    /// QR decomposition
    fn qr(self) -> Result<(Self, Self), LinalgError>;
    /// LU decomposition
    fn lu(self) -> Result<(Self::Permutator, Self, Self), LinalgError>;
    /// permutate matrix (inplace)
    fn permutate(&mut self, p: &Self::Permutator);
    /// permutate matrix (outplace)
    fn permutated(mut self, p: &Self::Permutator) -> Self {
        self.permutate(p);
        self
    }
}

impl<A: MFloat> Matrix for Array<A, Ix2> {
    type Scalar = A;
    type Vector = Array<A, Ix1>;
    type Permutator = Vec<i32>;

    fn size(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }
    fn layout(&self) -> Result<Layout, StrideError> {
        let strides = self.strides();
        if min(strides[0], strides[1]) != 1 {
            return Err(StrideError {
                s0: strides[0],
                s1: strides[1],
            });;
        }
        if strides[0] < strides[1] {
            Ok(Layout::ColumnMajor)
        } else {
            Ok(Layout::RowMajor)
        }
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
    fn svd(self) -> Result<(Self, Self::Vector, Self), LinalgError> {
        let (n, m) = self.size();
        let layout = self.layout()?;
        let (u, s, vt) = ImplSVD::svd(layout, m, n, self.clone().into_raw_vec())?;
        let sv = Array::from_vec(s);
        let ua = Array::from_vec(u).into_shape((n, n)).unwrap();
        let va = Array::from_vec(vt).into_shape((m, m)).unwrap();
        match layout {
            Layout::RowMajor => Ok((ua, sv, va)),
            Layout::ColumnMajor => Ok((ua.reversed_axes(), sv, va.reversed_axes())),
        }
    }
    fn qr(self) -> Result<(Self, Self), LinalgError> {
        let (n, m) = self.size();
        let strides = self.strides();
        let k = min(n, m);
        let layout = self.layout()?;
        let (q, r) = ImplQR::qr(layout, m, n, self.clone().into_raw_vec())?;
        let (qa, ra) = if strides[0] < strides[1] {
            (Array::from_vec(q).into_shape((m, n)).unwrap().reversed_axes(),
             Array::from_vec(r).into_shape((m, n)).unwrap().reversed_axes())
        } else {
            (Array::from_vec(q).into_shape((n, m)).unwrap(), Array::from_vec(r).into_shape((n, m)).unwrap())
        };
        let qm = if m > k {
            let (qsl, _) = qa.view().split_at(Axis(1), k);
            qsl.to_owned()
        } else {
            qa
        };
        let mut rm = if n > k {
            let (rsl, _) = ra.view().split_at(Axis(0), k);
            rsl.to_owned()
        } else {
            ra
        };
        for ((i, j), val) in rm.indexed_iter_mut() {
            if i > j {
                *val = A::zero();
            }
        }
        Ok((qm, rm))
    }
    fn lu(self) -> Result<(Self::Permutator, Self, Self), LinalgError> {
        let (n, m) = self.size();
        let k = min(n, m);
        let (p, l) = ImplSolve::lu(self.layout()?, n, m, self.clone().into_raw_vec())?;
        let mut a = match self.layout()? {
            Layout::ColumnMajor => Array::from_vec(l).into_shape((m, n)).unwrap().reversed_axes(),
            Layout::RowMajor => Array::from_vec(l).into_shape((n, m)).unwrap(),
        };
        let mut lm = Array::zeros((n, k));
        for ((i, j), val) in lm.indexed_iter_mut() {
            if i > j {
                *val = a[(i, j)];
            } else if i == j {
                *val = A::one();
            }
        }
        for ((i, j), val) in a.indexed_iter_mut() {
            if i > j {
                *val = A::zero();
            }
        }
        let am = if n > k {
            a.slice(s![0..k as isize, ..]).to_owned()
        } else {
            a
        };
        Ok((p, lm, am))
    }
    fn permutate(&mut self, ipiv: &Self::Permutator) {
        let (_, m) = self.size();
        for (i, j_) in ipiv.iter().enumerate().rev() {
            let j = (j_ - 1) as usize;
            if i == j {
                continue;
            }
            for k in 0..m {
                self.swap((i, k), (j, k));
            }
        }
    }
}
