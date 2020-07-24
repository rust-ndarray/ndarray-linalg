//! Define traits wrapping LAPACK routines

pub mod cholesky;
pub mod eig;
pub mod eigh;
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

use super::error::*;
use super::types::*;

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

pub fn into_result<T>(return_code: i32, val: T) -> Result<T> {
    if return_code == 0 {
        Ok(val)
    } else {
        Err(LinalgError::Lapack { return_code })
    }
}

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
