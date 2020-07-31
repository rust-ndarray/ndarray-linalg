//! Linear Algebra eXtension (LAX)
//! ===============================
//!
//! ndarray-free safe Rust wrapper for LAPACK FFI
//!
//! Linear equation, Inverse matrix, Condition number
//! --------------------------------------------------
//!
//! As the property of $A$, several types of triangular factorization are used:
//!
//! - LU-decomposition for general matrix
//!   - $PA = LU$, where $L$ is lower matrix, $U$ is upper matrix, and $P$ is permutation matrix
//! - Bunch-Kaufman diagonal pivoting method for nonpositive-definite Hermitian matrix
//!   - $A = U D U^\dagger$, where $U$ is upper matrix,
//!     $D$ is Hermitian and block diagonal with 1-by-1 and 2-by-2 diagonal blocks.
//!
//! | matrix type                     | Triangler factorization (TRF) | Solve (TRS) | Inverse matrix (TRI) | Reciprocal condition number (CON) |
//! |:--------------------------------|:------------------------------|:------------|:---------------------|:----------------------------------|
//! | General (GE)                    | [lu]                          | [solve]     | [inv]                | [rcond]                           |
//! | Symmetric (SY) / Hermitian (HE) | [bk]                          | [solveh]    | [invh]               | -                                 |
//!
//! [lu]:    solve/trait.Solve_.html#tymethod.lu
//! [solve]: solve/trait.Solve_.html#tymethod.solve
//! [inv]:   solve/trait.Solve_.html#tymethod.inv
//! [rcond]: solve/trait.Solve_.html#tymethod.rcond
//!
//! [bk]:     solveh/trait.Solveh_.html#tymethod.bk
//! [solveh]: solveh/trait.Solveh_.html#tymethod.solveh
//! [invh]:   solveh/trait.Solveh_.html#tymethod.invh
//!
//! Eigenvalue Problem
//! -------------------
//!
//! Solve eigenvalue problem for a matrix $A$
//!
//! $$ Av_i = \lambda_i v_i $$
//!
//! or generalized eigenvalue problem
//!
//! $$ Av_i = \lambda_i B v_i $$
//!
//! | matrix type                     | Eigenvalue (EV) | Generalized Eigenvalue Problem (EG) |
//! |:--------------------------------|:----------------|:------------------------------------|
//! | General (GE)                    |[eig]            | -                                   |
//! | Symmetric (SY) / Hermitian (HE) |[eigh]           |[eigh_generalized]                   |
//!
//! [eig]:              eig/trait.Eig_.html#tymethod.eig
//! [eigh]:             eigh/trait.Eigh_.html#tymethod.eigh
//! [eigh_generalized]: eigh/trait.Eigh_.html#tymethod.eigh_generalized
//!
//! Singular Value Decomposition (SVD), Least square problem
//! ----------------------------------------------------------
//!
//! | matrix type  | Singular Value Decomposition (SVD) | SVD with divided-and-conquer (SDD) | Least square problem (LSD) |
//! |:-------------|:-----------------------------------|:-----------------------------------|:---------------------------|
//! | General (GE) | [svd]                              | [svddc]                            | [least_squares]            |
//!
//! [svd]:   svd/trait.SVD_.html#tymethod.svd
//! [svddc]: svddck/trait.SVDDC_.html#tymethod.svddc
//! [least_squares]: least_squares/trait.LeastSquaresSvdDivideConquer_.html#tymethod.least_squares

#[cfg(any(feature = "intel-mkl-system", feature = "intel-mkl-static"))]
extern crate intel_mkl_src as _src;

#[cfg(any(feature = "openblas-system", feature = "openblas-static"))]
extern crate openblas_src as _src;

#[cfg(any(feature = "netlib-system", feature = "netlib-static"))]
extern crate netlib_src as _src;

pub mod error;
pub mod layout;

mod cholesky;
mod eig;
mod eigh;
mod eigh_generalized;
mod least_squares;
mod opnorm;
mod qr;
mod rcond;
mod solve;
mod solveh;
mod svd;
mod svddc;
mod triangular;
mod tridiagonal;

pub use self::eig::*;
pub use self::eigh::*;
pub use self::eigh_generalized::*;
pub use self::least_squares::*;
pub use self::opnorm::*;
pub use self::qr::*;
pub use self::rcond::*;
pub use self::solve::*;
pub use self::solveh::*;
pub use self::svd::*;
pub use self::svddc::*;
pub use self::triangular::*;
pub use self::tridiagonal::*;

use self::{cholesky::*, error::*, layout::*};
use cauchy::*;

pub type Pivot = Vec<i32>;

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

/// Upper/Lower specification for seveal usages
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum UPLO {
    Upper = b'U',
    Lower = b'L',
}

impl UPLO {
    pub fn t(self) -> Self {
        match self {
            UPLO::Upper => UPLO::Lower,
            UPLO::Lower => UPLO::Upper,
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum Transpose {
    No = b'N',
    Transpose = b'T',
    Hermite = b'C',
}

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum NormType {
    One = b'O',
    Infinity = b'I',
    Frobenius = b'F',
}

impl NormType {
    pub fn transpose(self) -> Self {
        match self {
            NormType::One => NormType::Infinity,
            NormType::Infinity => NormType::One,
            NormType::Frobenius => NormType::Frobenius,
        }
    }
}

/// Create a vector without initialization
///
/// Safety
/// ------
/// - Memory is not initialized. Do not read the memory before write.
///
unsafe fn vec_uninit<T: Sized>(n: usize) -> Vec<T> {
    let mut v = Vec::with_capacity(n);
    v.set_len(n);
    v
}
