//! Implement linear solver using LU decomposition
//! for tridiagonal matrix

use lapacke;
use ndarray::*;
use num_traits::Zero;

use super::NormType;
use super::{into_result, Pivot, Transpose};

use crate::error::*;
use crate::layout::MatrixLayout;
use crate::tridiagonal::{LUFactorizedTriDiagonal, TriDiagonal};
use crate::types::*;

/// Wraps `*gttrf`, `*gtcon` and `*gttrs`
pub trait TriDiagonal_: Scalar + Sized {
    /// Computes the LU factorization of a tridiagonal `m x n` matrix `a` using
    /// partial pivoting with row interchanges.
    unsafe fn lu_tridiagonal(a: &mut TriDiagonal<Self>) -> Result<(Array1<Self>, Pivot)>;
    /// Estimates the the reciprocal of the condition number of the tridiagonal matrix in 1-norm.
    unsafe fn rcond_tridiagonal(lu: &LUFactorizedTriDiagonal<Self>) -> Result<Self::Real>;
    unsafe fn solve_tridiagonal(
        lu: &LUFactorizedTriDiagonal<Self>,
        bl: MatrixLayout,
        t: Transpose,
        b: &mut [Self],
    ) -> Result<()>;
}

macro_rules! impl_tridiagonal {
    ($scalar:ty, $gttrf:path, $gtcon:path, $gttrs:path) => {
        impl TriDiagonal_ for $scalar {
            unsafe fn lu_tridiagonal(a: &mut TriDiagonal<Self>) -> Result<(Array1<Self>, Pivot)> {
                let (n, _) = a.l.size();
                let dl = a.dl.as_slice_mut().unwrap();
                let d = a.d.as_slice_mut().unwrap();
                let du = a.du.as_slice_mut().unwrap();
                let mut du2 = vec![Zero::zero(); (n - 2) as usize];
                let mut ipiv = vec![0; n as usize];
                let info = $gttrf(n, dl, d, du, &mut du2, &mut ipiv);
                into_result(info, (arr1(&du2), ipiv))
            }

            unsafe fn rcond_tridiagonal(lu: &LUFactorizedTriDiagonal<Self>) -> Result<Self::Real> {
                let (n, _) = lu.a.l.size();
                let dl = lu.a.dl.as_slice().unwrap();
                let d = lu.a.d.as_slice().unwrap();
                let du = lu.a.du.as_slice().unwrap();
                let du2 = lu.du2.as_slice().unwrap();
                let ipiv = &lu.ipiv;
                let anorm = lu.a.n1;
                let mut rcond = Self::Real::zero();
                let info = $gtcon(
                    NormType::One as u8,
                    n,
                    dl,
                    d,
                    du,
                    du2,
                    ipiv,
                    anorm,
                    &mut rcond,
                );
                into_result(info, rcond)
            }

            unsafe fn solve_tridiagonal(
                lu: &LUFactorizedTriDiagonal<Self>,
                bl: MatrixLayout,
                t: Transpose,
                b: &mut [Self],
            ) -> Result<()> {
                let (n, _) = lu.a.l.size();
                let (_, nrhs) = bl.size();
                let dl = lu.a.dl.as_slice().unwrap();
                let d = lu.a.d.as_slice().unwrap();
                let du = lu.a.du.as_slice().unwrap();
                let du2 = lu.du2.as_slice().unwrap();
                let ipiv = &lu.ipiv;
                let ldb = bl.lda();
                let info = $gttrs(
                    lu.a.l.lapacke_layout(),
                    t as u8,
                    n,
                    nrhs,
                    dl,
                    d,
                    du,
                    du2,
                    ipiv,
                    b,
                    ldb,
                );
                into_result(info, ())
            }
        }
    };
} // impl_tridiagonal!

impl_tridiagonal!(f64, lapacke::dgttrf, lapacke::dgtcon, lapacke::dgttrs);
impl_tridiagonal!(f32, lapacke::sgttrf, lapacke::sgtcon, lapacke::sgttrs);
impl_tridiagonal!(c64, lapacke::zgttrf, lapacke::zgtcon, lapacke::zgttrs);
impl_tridiagonal!(c32, lapacke::cgttrf, lapacke::cgtcon, lapacke::cgttrs);
