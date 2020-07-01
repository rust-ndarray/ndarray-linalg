//! Implement linear solver using LU decomposition
//! for tridiagonal matrix

use super::*;
use crate::{error::*, layout::MatrixLayout};
use cauchy::*;
use num_traits::Zero;
use std::ops::{Index, IndexMut};

/// Represents a tridiagonal matrix as 3 one-dimensional vectors.
///
/// ```text
/// [d0, u1,  0,   ...,       0,
///  l1, d1, u2,            ...,
///   0, l2, d2,
///  ...           ...,  u{n-1},
///   0,  ...,  l{n-1},  d{n-1},]
/// ```
#[derive(Clone, PartialEq)]
pub struct Tridiagonal<A: Scalar> {
    /// layout of raw matrix
    pub l: MatrixLayout,
    /// (n-1) sub-diagonal elements of matrix.
    pub dl: Vec<A>,
    /// (n) diagonal elements of matrix.
    pub d: Vec<A>,
    /// (n-1) super-diagonal elements of matrix.
    pub du: Vec<A>,
}

impl<A: Scalar> Tridiagonal<A> {
    fn opnorm_one(&self) -> A::Real {
        let mut col_sum: Vec<A::Real> = self.d.iter().map(|val| val.abs()).collect();
        for i in 0..col_sum.len() {
            if i < self.dl.len() {
                col_sum[i] += self.dl[i].abs();
            }
            if i > 0 {
                col_sum[i] += self.du[i - 1].abs();
            }
        }
        let mut max = A::Real::zero();
        for &val in &col_sum {
            if max < val {
                max = val;
            }
        }
        max
    }
}

/// Represents the LU factorization of a tridiagonal matrix `A` as `A = P*L*U`.
#[derive(Clone, PartialEq)]
pub struct LUFactorizedTridiagonal<A: Scalar> {
    /// A tridiagonal matrix which consists of
    /// - l : layout of raw matrix
    /// - dl: (n-1) multipliers that define the matrix L.
    /// - d : (n) diagonal elements of the upper triangular matrix U.
    /// - du: (n-1) elements of the first super-diagonal of U.
    pub a: Tridiagonal<A>,
    /// (n-2) elements of the second super-diagonal of U.
    pub du2: Vec<A>,
    /// The pivot indices that define the permutation matrix `P`.
    pub ipiv: Pivot,

    a_opnorm_one: A::Real,
}

impl<A: Scalar> Index<(i32, i32)> for Tridiagonal<A> {
    type Output = A;
    #[inline]
    fn index(&self, (row, col): (i32, i32)) -> &A {
        let (n, _) = self.l.size();
        assert!(
            std::cmp::max(row, col) < n,
            "ndarray: index {:?} is out of bounds for array of shape {}",
            [row, col],
            n
        );
        match row - col {
            0 => &self.d[row as usize],
            1 => &self.dl[col as usize],
            -1 => &self.du[row as usize],
            _ => panic!(
                "ndarray-linalg::tridiagonal: index {:?} is not tridiagonal element",
                [row, col]
            ),
        }
    }
}

impl<A: Scalar> Index<[i32; 2]> for Tridiagonal<A> {
    type Output = A;
    #[inline]
    fn index(&self, [row, col]: [i32; 2]) -> &A {
        &self[(row, col)]
    }
}

impl<A: Scalar> IndexMut<(i32, i32)> for Tridiagonal<A> {
    #[inline]
    fn index_mut(&mut self, (row, col): (i32, i32)) -> &mut A {
        let (n, _) = self.l.size();
        assert!(
            std::cmp::max(row, col) < n,
            "ndarray: index {:?} is out of bounds for array of shape {}",
            [row, col],
            n
        );
        match row - col {
            0 => &mut self.d[row as usize],
            1 => &mut self.dl[col as usize],
            -1 => &mut self.du[row as usize],
            _ => panic!(
                "ndarray-linalg::tridiagonal: index {:?} is not tridiagonal element",
                [row, col]
            ),
        }
    }
}

impl<A: Scalar> IndexMut<[i32; 2]> for Tridiagonal<A> {
    #[inline]
    fn index_mut(&mut self, [row, col]: [i32; 2]) -> &mut A {
        &mut self[(row, col)]
    }
}

/// Wraps `*gttrf`, `*gtcon` and `*gttrs`
pub trait Tridiagonal_: Scalar + Sized {
    /// Computes the LU factorization of a tridiagonal `m x n` matrix `a` using
    /// partial pivoting with row interchanges.
    unsafe fn lu_tridiagonal(a: Tridiagonal<Self>) -> Result<LUFactorizedTridiagonal<Self>>;

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
                mut a: Tridiagonal<Self>,
            ) -> Result<LUFactorizedTridiagonal<Self>> {
                let (n, _) = a.l.size();
                let mut du2 = vec![Zero::zero(); (n - 2) as usize];
                let mut ipiv = vec![0; n as usize];
                // We have to calc one-norm before LU factorization
                let a_opnorm_one = a.opnorm_one();
                $gttrf(n, &mut a.dl, &mut a.d, &mut a.du, &mut du2, &mut ipiv)
                    .as_lapack_result()?;
                Ok(LUFactorizedTridiagonal {
                    a,
                    du2,
                    ipiv,
                    a_opnorm_one,
                })
            }

            unsafe fn rcond_tridiagonal(lu: &LUFactorizedTridiagonal<Self>) -> Result<Self::Real> {
                let (n, _) = lu.a.l.size();
                let ipiv = &lu.ipiv;
                let mut rcond = Self::Real::zero();
                $gtcon(
                    NormType::One as u8,
                    n,
                    &lu.a.dl,
                    &lu.a.d,
                    &lu.a.du,
                    &lu.du2,
                    ipiv,
                    lu.a_opnorm_one,
                    &mut rcond,
                )
                .as_lapack_result()?;
                Ok(rcond)
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
                $gttrs(
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
                )
                .as_lapack_result()?;
                Ok(())
            }
        }
    };
} // impl_tridiagonal!

impl_tridiagonal!(f64, lapacke::dgttrf, lapacke::dgtcon, lapacke::dgttrs);
impl_tridiagonal!(f32, lapacke::sgttrf, lapacke::sgtcon, lapacke::sgttrs);
impl_tridiagonal!(c64, lapacke::zgttrf, lapacke::zgtcon, lapacke::zgttrs);
impl_tridiagonal!(c32, lapacke::cgttrf, lapacke::cgtcon, lapacke::cgttrs);
