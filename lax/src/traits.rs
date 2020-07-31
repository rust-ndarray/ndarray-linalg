use crate::{cholesky::*, error::*, layout::*, *};
use cauchy::*;

/// Trait for primitive types which implements LAPACK subroutines, i.e. [f32], [f64], [c32], and [c64]
///
/// [f32]: https://doc.rust-lang.org/std/primitive.f32.html
/// [f64]: https://doc.rust-lang.org/std/primitive.f64.html
/// [c32]: https://docs.rs/num-complex/0.2.4/num_complex/type.Complex32.html
/// [c64]: https://docs.rs/num-complex/0.2.4/num_complex/type.Complex64.html
pub trait Lapack:
    OperatorNorm_
    + QR_
    + SVD_
    + SVDDC_
    + Solve_
    + Solveh_
    + Eig_
    + Triangular_
    + Tridiagonal_
    + Rcond_
    + LeastSquaresSvdDivideConquer_
{
    /// Cholesky factorization for symmetric positive denite matrix $A$:
    ///
    /// $$ A = U^T U $$
    ///
    /// if `uplo == UPLO::Upper`, and
    ///
    /// $$ A = L L^T $$
    ///
    /// if `uplo == UPLO::Lower`,
    /// where $U$ is an upper triangular matrix and $L$ is lower triangular.
    ///
    /// **Only the portion of `a` corresponding to `UPLO` is written**.
    ///
    /// LAPACK routines
    /// ----------------
    /// - [spotrf](http://www.netlib.org/lapack/explore-html/d8/db2/group__real_p_ocomputational_gaaf31db7ab15b4f4ba527a3d31a15a58e.html)
    /// - [dpotrf](http://www.netlib.org/lapack/explore-html/d1/d7a/group__double_p_ocomputational_ga2f55f604a6003d03b5cd4a0adcfb74d6.html)
    /// - [cpotrf](http://www.netlib.org/lapack/explore-html/d6/df6/group__complex_p_ocomputational_ga4e85f48dbd837ccbbf76aa077f33de19.html)
    /// - [zpotrf](http://www.netlib.org/lapack/explore-html/d3/d8d/group__complex16_p_ocomputational_ga93e22b682170873efb50df5a79c5e4eb.html)
    fn cholesky(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<()>;

    /// Inverse of a real symmetric positive definite matrix $A$ using the Cholesky factorization
    ///
    /// LAPACK routines
    /// ----------------
    /// - [spotri](http://www.netlib.org/lapack/explore-html/d8/db2/group__real_p_ocomputational_ga4c381894bb34b1583fcc0dceafc5bea1.html)
    /// - [dpotri](http://www.netlib.org/lapack/explore-html/d1/d7a/group__double_p_ocomputational_ga9dfc04beae56a3b1c1f75eebc838c14c.html)
    /// - [cpotri](http://www.netlib.org/lapack/explore-html/d6/df6/group__complex_p_ocomputational_ga52b8da4d314abefaee93dd5c1ed7739e.html)
    /// - [zpotri](http://www.netlib.org/lapack/explore-html/d3/d8d/group__complex16_p_ocomputational_gaf37e3b8bbacd3332e83ffb3f1018bcf1.html)
    fn inv_cholesky(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<()>;

    /// Solves a system of linear equations $Ax = b$
    /// with a symmetric positive definite matrix $A$ using the Cholesky factorization
    ///
    /// LAPACK routines
    /// ----------------
    /// - [spotrs](http://www.netlib.org/lapack/explore-html/d8/db2/group__real_p_ocomputational_gaf5cc1531aa5ffe706533fbca343d55dd.html)
    /// - [dpotrs](http://www.netlib.org/lapack/explore-html/d1/d7a/group__double_p_ocomputational_ga167aa0166c4ce726385f65e4ab05e7c1.html)
    /// - [cpotrs](http://www.netlib.org/lapack/explore-html/d6/df6/group__complex_p_ocomputational_gad9052b4b70569dfd6e8943971c9b38b2.html)
    /// - [zpotrs](http://www.netlib.org/lapack/explore-html/d3/d8d/group__complex16_p_ocomputational_gaa2116ea574b01efda584dff0b74c9fcd.html)
    fn solve_cholesky(l: MatrixLayout, uplo: UPLO, a: &[Self], b: &mut [Self]) -> Result<()>;

    fn eigh(
        calc_eigenvec: bool,
        layout: MatrixLayout,
        uplo: UPLO,
        a: &mut [Self],
    ) -> Result<Vec<Self::Real>>;

    fn eigh_generalized(
        calc_eigenvec: bool,
        layout: MatrixLayout,
        uplo: UPLO,
        a: &mut [Self],
        b: &mut [Self],
    ) -> Result<Vec<Self::Real>>;
}

macro_rules! impl_lapack {
    ($scalar:ty) => {
        impl Lapack for $scalar {
            fn cholesky(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<()> {
                Cholesky::cholesky(l, uplo, a)
            }

            fn inv_cholesky(l: MatrixLayout, uplo: UPLO, a: &mut [Self]) -> Result<()> {
                Cholesky::inv_cholesky(l, uplo, a)
            }

            fn solve_cholesky(
                l: MatrixLayout,
                uplo: UPLO,
                a: &[Self],
                b: &mut [Self],
            ) -> Result<()> {
                Cholesky::solve_cholesky(l, uplo, a, b)
            }

            fn eigh(
                calc_eigenvec: bool,
                layout: MatrixLayout,
                uplo: UPLO,
                a: &mut [Self],
            ) -> Result<Vec<Self::Real>> {
                let mut work: EighWork<Self> = Eigh::eigh_work(calc_eigenvec, layout, uplo)?;
                let eigs = Eigh::eigh_calc(&mut work, a)?;
                Ok(eigs.into())
            }

            fn eigh_generalized(
                calc_eigenvec: bool,
                layout: MatrixLayout,
                uplo: UPLO,
                a: &mut [Self],
                b: &mut [Self],
            ) -> Result<Vec<Self::Real>> {
                let mut work: EighGeneralizedWork<Self> =
                    EighGeneralized::eigh_generalized_work(calc_eigenvec, layout, uplo)?;
                let eigs = EighGeneralized::eigh_generalized_calc(&mut work, a, b)?;
                Ok(eigs.into())
            }
        }
    };
}

impl_lapack!(f32);
impl_lapack!(f64);
impl_lapack!(c32);
impl_lapack!(c64);
