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

extern crate blas_src;
extern crate lapack_src;

pub mod cholesky;
pub mod eig;
pub mod eigh;
pub mod error;
pub mod layout;
pub mod least_squares;
pub mod opnorm;
pub mod qr;
pub mod solve;
pub mod solveh;
pub mod svd;
pub mod svddc;
pub mod triangular;
pub mod tridiagonal;

pub use self::cholesky::*;
pub use self::eig::*;
pub use self::eigh::*;
pub use self::least_squares::*;
pub use self::opnorm::*;
pub use self::qr::*;
pub use self::solve::*;
pub use self::solveh::*;
pub use self::svd::*;
pub use self::svddc::*;
pub use self::triangular::*;
pub use self::tridiagonal::*;

use cauchy::*;

pub type Pivot = Vec<i32>;

/// Trait for primitive types which implements LAPACK subroutines
pub trait Lapack:
    OperatorNorm_
    + QR_
    + SVD_
    + SVDDC_
    + Solve_
    + Solveh_
    + Cholesky_
    + Eig_
    + Eigh_
    + Triangular_
    + Tridiagonal_
    + LeastSquaresSvdDivideConquer_
{
}

impl Lapack for f32 {}
impl Lapack for f64 {}
impl Lapack for c32 {}
impl Lapack for c64 {}

/// Upper/Lower specification for seveal usages
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum UPLO {
    Upper = b'U',
    Lower = b'L',
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
    pub(crate) fn transpose(self) -> Self {
        match self {
            NormType::One => NormType::Infinity,
            NormType::Infinity => NormType::One,
            NormType::Frobenius => NormType::Frobenius,
        }
    }
}
