use super::*;
use crate::{error::*, layout::*};
use cauchy::*;

#[cfg_attr(doc, katexit::katexit)]
/// Solve symmetric/hermite positive-definite linear equations using Cholesky decomposition
///
/// For a given positive definite matrix $A$,
/// Cholesky decomposition is described as $A = U^T U$ or $A = LL^T$ where
///
/// - $L$ is lower matrix
/// - $U$ is upper matrix
///
/// This is designed as two step computation according to LAPACK API
///
/// 1. Factorize input matrix $A$ into $L$ or $U$
/// 2. Solve linear equation $Ax = b$ or compute inverse matrix $A^{-1}$
///    using $U$ or $L$.
pub trait Cholesky_: Sized {
    /// Compute Cholesky decomposition $A = U^T U$ or $A = L L^T$ according to [UPLO]
    ///
    /// LAPACK correspondance
    /// ----------------------
    ///
    /// | f32    | f64    | c32    | c64    |
    /// |:-------|:-------|:-------|:-------|
    /// | spotrf | dpotrf | cpotrf | zpotrf |
    ///
    fn cholesky(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<()>;

    /// Compute inverse matrix $A^{-1}$ using $U$ or $L$
    ///
    /// LAPACK correspondance
    /// ----------------------
    ///
    /// | f32    | f64    | c32    | c64    |
    /// |:-------|:-------|:-------|:-------|
    /// | spotri | dpotri | cpotri | zpotri |
    ///
    fn inv_cholesky(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<()>;

    /// Solve linear equation $Ax = b$ using $U$ or $L$
    ///
    /// LAPACK correspondance
    /// ----------------------
    ///
    /// | f32    | f64    | c32    | c64    |
    /// |:-------|:-------|:-------|:-------|
    /// | spotrs | dpotrs | cpotrs | zpotrs |
    ///
    fn solve_cholesky(l: MatrixLayout, uplo: UPLO, a: &[Self], b: &mut [Self]) -> Result<()>;
}

macro_rules! impl_cholesky {
    ($scalar:ty, $trf:path, $tri:path, $trs:path) => {
        impl Cholesky_ for $scalar {
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
} // end macro_rules

impl_cholesky!(
    f64,
    lapack_sys::dpotrf_,
    lapack_sys::dpotri_,
    lapack_sys::dpotrs_
);
impl_cholesky!(
    f32,
    lapack_sys::spotrf_,
    lapack_sys::spotri_,
    lapack_sys::spotrs_
);
impl_cholesky!(
    c64,
    lapack_sys::zpotrf_,
    lapack_sys::zpotri_,
    lapack_sys::zpotrs_
);
impl_cholesky!(
    c32,
    lapack_sys::cpotrf_,
    lapack_sys::cpotri_,
    lapack_sys::cpotrs_
);
