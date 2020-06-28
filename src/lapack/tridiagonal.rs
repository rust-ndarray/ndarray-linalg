//! Implement linear solver using LU decomposition
//! for tridiagonal matrix

use num_traits::Zero;

use super::NormType;
use super::{into_result, Pivot, Transpose};

use crate::error::*;
use crate::layout::MatrixLayout;
use crate::opnorm::*;
use crate::tridiagonal::{LUFactorizedTridiagonal, Tridiagonal};
use crate::types::*;

/// Wraps `*gttrf`, `*gtcon` and `*gttrs`
pub trait Tridiagonal_: Scalar + Sized {
    /// Computes the LU factorization of a tridiagonal `m x n` matrix `a` using
    /// partial pivoting with row interchanges.
    unsafe fn lu_tridiagonal(a: &mut Tridiagonal<Self>) -> Result<(Vec<Self>, Self::Real, Pivot)>;
    /// Estimates the the reciprocal of the condition number of the tridiagonal matrix in 1-norm.
    unsafe fn rcond_tridiagonal(lu: &LUFactorizedTridiagonal<Self>) -> Result<Self::Real>;
    unsafe fn solve_tridiagonal(
        lu: &LUFactorizedTridiagonal<Self>,
        bl: MatrixLayout,
        t: Transpose,
        b: &mut [Self],
    ) -> Result<()>;
}

macro_rules! impl_tridiagonal {
    ($scalar:ty, $gttrf:path, $gtcon:path, $gttrs:path) => {
        impl Tridiagonal_ for $scalar {
            unsafe fn lu_tridiagonal(
                a: &mut Tridiagonal<Self>,
            ) -> Result<(Vec<Self>, Self::Real, Pivot)> {
                let (n, _) = a.l.size();
                let anom = a.opnorm_one()?;
                let mut du2 = vec![Zero::zero(); (n - 2) as usize];
                let mut ipiv = vec![0; n as usize];
                let info = $gttrf(n, &mut a.dl, &mut a.d, &mut a.du, &mut du2, &mut ipiv);
                into_result(info, (du2, anom, ipiv))
            }

            unsafe fn rcond_tridiagonal(lu: &LUFactorizedTridiagonal<Self>) -> Result<Self::Real> {
                let (n, _) = lu.a.l.size();
                let ipiv = &lu.ipiv;
                let anorm = lu.anom;
                let mut rcond = Self::Real::zero();
                let info = $gtcon(
                    NormType::One as u8,
                    n,
                    &lu.a.dl,
                    &lu.a.d,
                    &lu.a.du,
                    &lu.du2,
                    ipiv,
                    anorm,
                    &mut rcond,
                );
                into_result(info, rcond)
            }

            unsafe fn solve_tridiagonal(
                lu: &LUFactorizedTridiagonal<Self>,
                bl: MatrixLayout,
                t: Transpose,
                b: &mut [Self],
            ) -> Result<()> {
                let (n, _) = lu.a.l.size();
                let (_, nrhs) = bl.size();
                let ipiv = &lu.ipiv;
                let ldb = bl.lda();
                let info = $gttrs(
                    lu.a.l.lapacke_layout(),
                    t as u8,
                    n,
                    nrhs,
                    &lu.a.dl,
                    &lu.a.d,
                    &lu.a.du,
                    &lu.du2,
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
