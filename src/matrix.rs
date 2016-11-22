//! Define trait for general matrix

use std::cmp::min;
use std::fmt::Debug;
use ndarray::prelude::*;
use ndarray::LinalgScalar;
use lapack::c::Layout;

use error::LapackError;
use qr::ImplQR;
use svd::ImplSVD;
use norm::ImplNorm;
use solve::ImplSolve;

/// Methods for general matrices
pub trait Matrix: Sized {
    type Scalar;
    type Vector;
    type Permutator;
    /// number of (rows, columns)
    fn size(&self) -> (usize, usize);
    /// Layout (C/Fortran) of matrix
    fn layout(&self) -> Layout;
    /// Operator norm for L-1 norm
    fn norm_1(&self) -> Self::Scalar;
    /// Operator norm for L-inf norm
    fn norm_i(&self) -> Self::Scalar;
    /// Frobenius norm
    fn norm_f(&self) -> Self::Scalar;
    /// singular-value decomposition (SVD)
    fn svd(self) -> Result<(Self, Self::Vector, Self), LapackError>;
    /// QR decomposition
    fn qr(self) -> Result<(Self, Self), LapackError>;
    /// LU decomposition
    fn lu(self) -> Result<(Self::Permutator, Self, Self), LapackError>;
    /// permutate matrix (inplace)
    fn permutate(&mut self, p: &Self::Permutator);
    /// permutate matrix (outplace)
    fn permutated(mut self, p: &Self::Permutator) -> Self {
        self.permutate(p);
        self
    }
}

impl<A> Matrix for Array<A, Ix2>
    where A: ImplQR + ImplSVD + ImplNorm + ImplSolve + LinalgScalar + Debug
{
    type Scalar = A;
    type Vector = Array<A, Ix>;
    type Permutator = Vec<i32>;

    fn size(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }
    fn layout(&self) -> Layout {
        let strides = self.strides();
        if strides[0] < strides[1] {
            Layout::ColumnMajor
        } else {
            Layout::RowMajor
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
        let (q, r) = if strides[0] < strides[1] {
            try!(ImplQR::qr(m, n, self.clone().into_raw_vec()))
        } else {
            try!(ImplQR::lq(n, m, self.clone().into_raw_vec()))
        };
        let (qa, ra) = if strides[0] < strides[1] {
            (Array::from_vec(q).into_shape((m, n)).unwrap().reversed_axes(),
             Array::from_vec(r).into_shape((m, n)).unwrap().reversed_axes())
        } else {
            (Array::from_vec(q).into_shape((n, m)).unwrap(),
             Array::from_vec(r).into_shape((n, m)).unwrap())
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
    fn lu(self) -> Result<(Self::Permutator, Self, Self), LapackError> {
        let (n, m) = self.size();
        println!("n={}, m={}", n, m);
        let k = min(n, m);
        let (p, mut a) = match self.layout() {
            Layout::ColumnMajor => {
                println!("ColumnMajor");
                let (p, l) = ImplSolve::lu(self.layout(), n, m, self.clone().into_raw_vec())?;
                (p, Array::from_vec(l).into_shape((m, n)).unwrap().reversed_axes())
            }
            Layout::RowMajor => {
                println!("RowMajor");
                let (p, l) = ImplSolve::lu(self.layout(), n, m, self.clone().into_raw_vec())?;
                (p, Array::from_vec(l).into_shape((n, m)).unwrap())
            }
        };
        println!("a (after LU) = \n{:?}", &a);
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
        println!("am = \n{:?}", am);
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
