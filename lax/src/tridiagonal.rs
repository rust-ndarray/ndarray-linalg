//! Implement linear solver using LU decomposition
//! for tridiagonal matrix

use crate::{error::*, layout::*, *};
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
    fn lu_tridiagonal(a: Tridiagonal<Self>) -> Result<LUFactorizedTridiagonal<Self>>;

    fn rcond_tridiagonal(lu: &LUFactorizedTridiagonal<Self>) -> Result<Self::Real>;

    fn solve_tridiagonal(
        lu: &LUFactorizedTridiagonal<Self>,
        bl: MatrixLayout,
        t: Transpose,
        b: &mut [Self],
    ) -> Result<()>;
}

macro_rules! impl_tridiagonal {
    (@real, $scalar:ty, $gttrf:path, $gtcon:path, $gttrs:path) => {
        impl_tridiagonal!(@body, $scalar, $gttrf, $gtcon, $gttrs, iwork);
    };
    (@complex, $scalar:ty, $gttrf:path, $gtcon:path, $gttrs:path) => {
        impl_tridiagonal!(@body, $scalar, $gttrf, $gtcon, $gttrs, );
    };
    (@body, $scalar:ty, $gttrf:path, $gtcon:path, $gttrs:path, $($iwork:ident)*) => {
        impl Tridiagonal_ for $scalar {
            fn lu_tridiagonal(mut a: Tridiagonal<Self>) -> Result<LUFactorizedTridiagonal<Self>> {
                let (n, _) = a.l.size();
                let mut du2 = unsafe { vec_uninit( (n - 2) as usize) };
                let mut ipiv = unsafe { vec_uninit( n as usize) };
                // We have to calc one-norm before LU factorization
                let a_opnorm_one = a.opnorm_one();
                let mut info = 0;
                unsafe { $gttrf(n, &mut a.dl, &mut a.d, &mut a.du, &mut du2, &mut ipiv, &mut info,) };
                info.as_lapack_result()?;
                Ok(LUFactorizedTridiagonal {
                    a,
                    du2,
                    ipiv,
                    a_opnorm_one,
                })
            }

            fn rcond_tridiagonal(lu: &LUFactorizedTridiagonal<Self>) -> Result<Self::Real> {
                let (n, _) = lu.a.l.size();
                let ipiv = &lu.ipiv;
                let mut work = unsafe { vec_uninit( 2 * n as usize) };
                $(
                let mut $iwork = unsafe { vec_uninit( n as usize) };
                )*
                let mut rcond = Self::Real::zero();
                let mut info = 0;
                unsafe {
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
                        &mut work,
                        $(&mut $iwork,)*
                        &mut info,
                    );
                }
                info.as_lapack_result()?;
                Ok(rcond)
            }

            fn solve_tridiagonal(
                lu: &LUFactorizedTridiagonal<Self>,
                b_layout: MatrixLayout,
                t: Transpose,
                b: &mut [Self],
            ) -> Result<()> {
                let (n, _) = lu.a.l.size();
                let ipiv = &lu.ipiv;
                // Transpose if b is C-continuous
                let mut b_t = None;
                let b_layout = match b_layout {
                    MatrixLayout::C { .. } => {
                        b_t = Some(unsafe { vec_uninit( b.len()) });
                        transpose(b_layout, b, b_t.as_mut().unwrap())
                    }
                    MatrixLayout::F { .. } => b_layout,
                };
                let (ldb, nrhs) = b_layout.size();
                let mut info = 0;
                unsafe {
                    $gttrs(
                        t as u8,
                        n,
                        nrhs,
                        &lu.a.dl,
                        &lu.a.d,
                        &lu.a.du,
                        &lu.du2,
                        ipiv,
                        b_t.as_mut().map(|v| v.as_mut_slice()).unwrap_or(b),
                        ldb,
                        &mut info,
                    );
                }
                info.as_lapack_result()?;
                if let Some(b_t) = b_t {
                    transpose(b_layout, &b_t, b);
                }
                Ok(())
            }
        }
    };
} // impl_tridiagonal!

impl_tridiagonal!(@real, f64, lapack::dgttrf, lapack::dgtcon, lapack::dgttrs);
impl_tridiagonal!(@real, f32, lapack::sgttrf, lapack::sgtcon, lapack::sgttrs);
impl_tridiagonal!(@complex, c64, lapack::zgttrf, lapack::zgtcon, lapack::zgttrs);
impl_tridiagonal!(@complex, c32, lapack::cgttrf, lapack::cgtcon, lapack::cgttrs);
