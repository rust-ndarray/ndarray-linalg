//! Define trait for general matrix

use std::cmp::min;
use ndarray::*;
use ndarray::DataMut;
use lapack::c::Layout;

use super::error::{LinalgError, StrideError};
use super::impls::solve::ImplSolve;

pub trait MFloat: ImplSolve + NdFloat {}
impl<A: ImplSolve + NdFloat> MFloat for A {}

/// Methods for general matrices
pub trait Matrix: Sized {
    type Scalar;
    type Vector;
    type Permutator;
    /// number of (rows, columns)
    fn size(&self) -> (usize, usize);
    /// Layout (C/Fortran) of matrix
    fn layout(&self) -> Result<Layout, StrideError>;
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

fn check_layout(strides: &[Ixs]) -> Result<Layout, StrideError> {
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

fn permutate<A: NdFloat, S>(mut a: &mut ArrayBase<S, Ix2>, ipiv: &Vec<i32>)
    where S: DataMut<Elem = A>
{
    let m = a.cols();
    for (i, j_) in ipiv.iter().enumerate().rev() {
        let j = (j_ - 1) as usize;
        if i == j {
            continue;
        }
        for k in 0..m {
            a.swap((i, k), (j, k));
        }
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
        check_layout(self.strides())
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
        permutate(self, ipiv);
    }
}

impl<A: MFloat> Matrix for RcArray<A, Ix2> {
    type Scalar = A;
    type Vector = RcArray<A, Ix1>;
    type Permutator = Vec<i32>;
    fn size(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }
    fn layout(&self) -> Result<Layout, StrideError> {
        check_layout(self.strides())
    }
    fn lu(self) -> Result<(Self::Permutator, Self, Self), LinalgError> {
        let (p, l, u) = self.into_owned().lu()?;
        Ok((p, l.into_shared(), u.into_shared()))
    }
    fn permutate(&mut self, ipiv: &Self::Permutator) {
        permutate(self, ipiv);
    }
}
