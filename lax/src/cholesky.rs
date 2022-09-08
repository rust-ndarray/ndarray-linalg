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
    /// | f32      | f64      | c32      | c64      |
    /// |:---------|:---------|:---------|:---------|
    /// | [spotrf] | [dpotrf] | [cpotrf] | [zpotrf] |
    ///
    /// [spotrf]: https://netlib.org/lapack/explore-html/d8/db2/group__real_p_ocomputational_gaaf31db7ab15b4f4ba527a3d31a15a58e.html
    /// [dpotrf]: https://netlib.org/lapack/explore-html/d1/d7a/group__double_p_ocomputational_ga2f55f604a6003d03b5cd4a0adcfb74d6.html
    /// [cpotrf]: https://netlib.org/lapack/explore-html/d8/d6c/group__variants_p_ocomputational_ga4e85f48dbd837ccbbf76aa077f33de19.html
    /// [zpotrf]: https://netlib.org/lapack/explore-html/d3/d8d/group__complex16_p_ocomputational_ga93e22b682170873efb50df5a79c5e4eb.html
    ///
    fn cholesky(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<()>;

    /// Compute inverse matrix $A^{-1}$ using $U$ or $L$
    ///
    /// LAPACK correspondance
    /// ----------------------
    ///
    /// | f32      | f64      | c32      | c64      |
    /// |:---------|:---------|:---------|:---------|
    /// | [spotri] | [dpotri] | [cpotri] | [zpotri] |
    ///
    /// [spotri]: https://netlib.org/lapack/explore-html/d8/db2/group__real_p_ocomputational_ga4c381894bb34b1583fcc0dceafc5bea1.html
    /// [dpotri]: https://netlib.org/lapack/explore-html/d1/d7a/group__double_p_ocomputational_ga9dfc04beae56a3b1c1f75eebc838c14c.html
    /// [cpotri]: https://netlib.org/lapack/explore-html/d6/df6/group__complex_p_ocomputational_ga52b8da4d314abefaee93dd5c1ed7739e.html
    /// [zpotri]: https://netlib.org/lapack/explore-html/d3/d8d/group__complex16_p_ocomputational_gaf37e3b8bbacd3332e83ffb3f1018bcf1.html
    ///
    fn inv_cholesky(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<()>;

    /// Solve linear equation $Ax = b$ using $U$ or $L$
    ///
    /// LAPACK correspondance
    /// ----------------------
    ///
    /// | f32      | f64      | c32      | c64      |
    /// |:---------|:---------|:---------|:---------|
    /// | [spotrs] | [dpotrs] | [cpotrs] | [zpotrs] |
    ///
    /// [spotrs]: https://netlib.org/lapack/explore-html/d8/db2/group__real_p_ocomputational_gaf5cc1531aa5ffe706533fbca343d55dd.html
    /// [dpotrs]: https://netlib.org/lapack/explore-html/d1/d7a/group__double_p_ocomputational_ga167aa0166c4ce726385f65e4ab05e7c1.html
    /// [cpotrs]: https://netlib.org/lapack/explore-html/d6/df6/group__complex_p_ocomputational_gad9052b4b70569dfd6e8943971c9b38b2.html
    /// [zpotrs]: https://netlib.org/lapack/explore-html/d3/d8d/group__complex16_p_ocomputational_gaa2116ea574b01efda584dff0b74c9fcd.html
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
