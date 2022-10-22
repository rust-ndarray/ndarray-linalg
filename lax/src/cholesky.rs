//! Factorize positive-definite symmetric/Hermitian matrices using Cholesky algorithm

use super::*;
use crate::{error::*, layout::*};
use cauchy::*;

/// Compute Cholesky decomposition according to [UPLO]
///
/// LAPACK correspondance
/// ----------------------
///
/// | f32    | f64    | c32    | c64    |
/// |:-------|:-------|:-------|:-------|
/// | spotrf | dpotrf | cpotrf | zpotrf |
///
pub trait CholeskyImpl: Scalar {
    fn cholesky(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<()>;
}

macro_rules! impl_cholesky_ {
    ($s:ty, $trf:path) => {
        impl CholeskyImpl for $s {
            fn cholesky(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<()> {
                let (n, _) = l.size();
                if matches!(l, MatrixLayout::C { .. }) {
                    square_transpose(l, a);
                }
                let mut info = 0;
                unsafe {
                    $trf(uplo.as_ptr(), &n, AsPtr::as_mut_ptr(a), &n, &mut info);
                }
                info.as_lapack_result()?;
                if matches!(l, MatrixLayout::C { .. }) {
                    square_transpose(l, a);
                }
                Ok(())
            }
        }
    };
}
impl_cholesky_!(c64, lapack_sys::zpotrf_);
impl_cholesky_!(c32, lapack_sys::cpotrf_);
impl_cholesky_!(f64, lapack_sys::dpotrf_);
impl_cholesky_!(f32, lapack_sys::spotrf_);

/// Compute inverse matrix using Cholesky factroization result
///
/// LAPACK correspondance
/// ----------------------
///
/// | f32    | f64    | c32    | c64    |
/// |:-------|:-------|:-------|:-------|
/// | spotri | dpotri | cpotri | zpotri |
///
pub trait InvCholeskyImpl: Scalar {
    fn inv_cholesky(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<()>;
}

macro_rules! impl_inv_cholesky {
    ($s:ty, $tri:path) => {
        impl InvCholeskyImpl for $s {
            fn inv_cholesky(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<()> {
                let (n, _) = l.size();
                if matches!(l, MatrixLayout::C { .. }) {
                    square_transpose(l, a);
                }
                let mut info = 0;
                unsafe {
                    $tri(uplo.as_ptr(), &n, AsPtr::as_mut_ptr(a), &l.lda(), &mut info);
                }
                info.as_lapack_result()?;
                if matches!(l, MatrixLayout::C { .. }) {
                    square_transpose(l, a);
                }
                Ok(())
            }
        }
    };
}
impl_inv_cholesky!(c64, lapack_sys::zpotri_);
impl_inv_cholesky!(c32, lapack_sys::cpotri_);
impl_inv_cholesky!(f64, lapack_sys::dpotri_);
impl_inv_cholesky!(f32, lapack_sys::spotri_);

/// Solve linear equation using Cholesky factroization result
///
/// LAPACK correspondance
/// ----------------------
///
/// | f32    | f64    | c32    | c64    |
/// |:-------|:-------|:-------|:-------|
/// | spotrs | dpotrs | cpotrs | zpotrs |
///
pub trait SolveCholeskyImpl: Scalar {
    fn solve_cholesky(l: MatrixLayout, uplo: UPLO, a: &[Self], b: &mut [Self]) -> Result<()>;
}

macro_rules! impl_solve_cholesky {
    ($s:ty, $trs:path) => {
        impl SolveCholeskyImpl for $s {
            fn solve_cholesky(
                l: MatrixLayout,
                mut uplo: UPLO,
                a: &[Self],
                b: &mut [Self],
            ) -> Result<()> {
                let (n, _) = l.size();
                let nrhs = 1;
                let mut info = 0;
                if matches!(l, MatrixLayout::C { .. }) {
                    uplo = uplo.t();
                    for val in b.iter_mut() {
                        *val = val.conj();
                    }
                }
                unsafe {
                    $trs(
                        uplo.as_ptr(),
                        &n,
                        &nrhs,
                        AsPtr::as_ptr(a),
                        &l.lda(),
                        AsPtr::as_mut_ptr(b),
                        &n,
                        &mut info,
                    );
                }
                info.as_lapack_result()?;
                if matches!(l, MatrixLayout::C { .. }) {
                    for val in b.iter_mut() {
                        *val = val.conj();
                    }
                }
                Ok(())
            }
        }
    };
}
impl_solve_cholesky!(c64, lapack_sys::zpotrs_);
impl_solve_cholesky!(c32, lapack_sys::cpotrs_);
impl_solve_cholesky!(f64, lapack_sys::dpotrs_);
impl_solve_cholesky!(f32, lapack_sys::spotrs_);
