//! Implement linear solver using LU decomposition
//! for tridiagonal matrix

mod matrix;
mod solve;

pub use matrix::*;
pub use solve::*;

use crate::{error::*, layout::*, *};
use cauchy::*;
use num_traits::Zero;

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
                let mut du2 = vec_uninit( (n - 2) as usize);
                let mut ipiv = vec_uninit( n as usize);
                // We have to calc one-norm before LU factorization
                let a_opnorm_one = a.opnorm_one();
                let mut info = 0;
                unsafe {
                    $gttrf(
                        &n,
                        AsPtr::as_mut_ptr(&mut a.dl),
                        AsPtr::as_mut_ptr(&mut a.d),
                        AsPtr::as_mut_ptr(&mut a.du),
                        AsPtr::as_mut_ptr(&mut du2),
                        AsPtr::as_mut_ptr(&mut ipiv),
                        &mut info,
                    )
                };
                info.as_lapack_result()?;
                let du2 = unsafe { du2.assume_init() };
                let ipiv = unsafe { ipiv.assume_init() };
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
                let mut work: Vec<MaybeUninit<Self>> = vec_uninit(2 * n as usize);
                $(
                let mut $iwork: Vec<MaybeUninit<i32>> = vec_uninit(n as usize);
                )*
                let mut rcond = Self::Real::zero();
                let mut info = 0;
                unsafe {
                    $gtcon(
                        NormType::One.as_ptr(),
                        &n,
                        AsPtr::as_ptr(&lu.a.dl),
                        AsPtr::as_ptr(&lu.a.d),
                        AsPtr::as_ptr(&lu.a.du),
                        AsPtr::as_ptr(&lu.du2),
                        ipiv.as_ptr(),
                        &lu.a_opnorm_one,
                        &mut rcond,
                        AsPtr::as_mut_ptr(&mut work),
                        $(AsPtr::as_mut_ptr(&mut $iwork),)*
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
                        let (layout, t) = transpose(b_layout, b);
                        b_t = Some(t);
                        layout
                    }
                    MatrixLayout::F { .. } => b_layout,
                };
                let (ldb, nrhs) = b_layout.size();
                let mut info = 0;
                unsafe {
                    $gttrs(
                        t.as_ptr(),
                        &n,
                        &nrhs,
                        AsPtr::as_ptr(&lu.a.dl),
                        AsPtr::as_ptr(&lu.a.d),
                        AsPtr::as_ptr(&lu.a.du),
                        AsPtr::as_ptr(&lu.du2),
                        ipiv.as_ptr(),
                        AsPtr::as_mut_ptr(b_t.as_mut().map(|v| v.as_mut_slice()).unwrap_or(b)),
                        &ldb,
                        &mut info,
                    );
                }
                info.as_lapack_result()?;
                if let Some(b_t) = b_t {
                    transpose_over(b_layout, &b_t, b);
                }
                Ok(())
            }
        }
    };
} // impl_tridiagonal!

impl_tridiagonal!(@real, f64, lapack_sys::dgttrf_, lapack_sys::dgtcon_, lapack_sys::dgttrs_);
impl_tridiagonal!(@real, f32, lapack_sys::sgttrf_, lapack_sys::sgtcon_, lapack_sys::sgttrs_);
impl_tridiagonal!(@complex, c64, lapack_sys::zgttrf_, lapack_sys::zgtcon_, lapack_sys::zgttrs_);
impl_tridiagonal!(@complex, c32, lapack_sys::cgttrf_, lapack_sys::cgtcon_, lapack_sys::cgttrs_);
