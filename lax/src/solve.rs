//! Solve linear problem using LU decomposition

use crate::{error::*, layout::MatrixLayout, *};
use cauchy::*;
use num_traits::{ToPrimitive, Zero};

pub trait Solve_: Scalar + Sized {
    /// Computes the LU factorization of a general `m x n` matrix `a` using
    /// partial pivoting with row interchanges.
    ///
    /// $ PA = LU $
    ///
    /// Error
    /// ------
    /// - `LapackComputationalFailure { return_code }` when the matrix is singular
    ///   - Division by zero will occur if it is used to solve a system of equations
    ///     because `U[(return_code-1, return_code-1)]` is exactly zero.
    fn lu(l: MatrixLayout, a: &mut [Self]) -> Result<Pivot>;

    fn inv(l: MatrixLayout, a: &mut [Self], p: &Pivot) -> Result<()>;

    fn solve(l: MatrixLayout, t: Transpose, a: &[Self], p: &Pivot, b: &mut [Self]) -> Result<()>;
}

macro_rules! impl_solve {
    ($scalar:ty, $getrf:path, $getri:path, $getrs:path) => {
        impl Solve_ for $scalar {
            fn lu(l: MatrixLayout, a: &mut [Self]) -> Result<Pivot> {
                let (row, col) = l.size();
                assert_eq!(a.len() as i32, row * col);
                if row == 0 || col == 0 {
                    // Do nothing for empty matrix
                    return Ok(Vec::new());
                }
                let k = ::std::cmp::min(row, col);
                let mut ipiv = unsafe { vec_uninit(k as usize) };
                let mut info = 0;
                unsafe { $getrf(l.lda(), l.len(), a, l.lda(), &mut ipiv, &mut info) };
                info.as_lapack_result()?;
                Ok(ipiv)
            }

            fn inv(l: MatrixLayout, a: &mut [Self], ipiv: &Pivot) -> Result<()> {
                let (n, _) = l.size();

                // calc work size
                let mut info = 0;
                let mut work_size = [Self::zero()];
                unsafe { $getri(n, a, l.lda(), ipiv, &mut work_size, -1, &mut info) };
                info.as_lapack_result()?;

                // actual
                let lwork = work_size[0].to_usize().unwrap();
                let mut work = unsafe { vec_uninit(lwork) };
                unsafe {
                    $getri(
                        l.len(),
                        a,
                        l.lda(),
                        ipiv,
                        &mut work,
                        lwork as i32,
                        &mut info,
                    )
                };
                info.as_lapack_result()?;

                Ok(())
            }

            fn solve(
                l: MatrixLayout,
                t: Transpose,
                a: &[Self],
                ipiv: &Pivot,
                b: &mut [Self],
            ) -> Result<()> {
                let t = match l {
                    MatrixLayout::C { .. } => match t {
                        Transpose::No => Transpose::Transpose,
                        Transpose::Transpose | Transpose::Hermite => Transpose::No,
                    },
                    _ => t,
                };
                let (n, _) = l.size();
                let nrhs = 1;
                let ldb = l.lda();
                let mut info = 0;
                unsafe { $getrs(t as u8, n, nrhs, a, l.lda(), ipiv, b, ldb, &mut info) };
                info.as_lapack_result()?;
                Ok(())
            }
        }
    };
} // impl_solve!

impl_solve!(f64, lapack::dgetrf, lapack::dgetri, lapack::dgetrs);
impl_solve!(f32, lapack::sgetrf, lapack::sgetri, lapack::sgetrs);
impl_solve!(c64, lapack::zgetrf, lapack::zgetri, lapack::zgetrs);
impl_solve!(c32, lapack::cgetrf, lapack::cgetri, lapack::cgetrs);
