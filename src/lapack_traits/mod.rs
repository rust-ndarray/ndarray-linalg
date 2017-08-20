//! Define traits wrapping LAPACK routines

pub mod opnorm;
pub mod qr;
pub mod svd;
pub mod solve;
pub mod solveh;
pub mod cholesky;
pub mod eigh;
pub mod triangular;

pub use self::cholesky::*;
pub use self::eigh::*;
pub use self::opnorm::*;
pub use self::qr::*;
pub use self::solve::*;
pub use self::solveh::*;
pub use self::svd::*;
pub use self::triangular::*;

use super::error::*;
use super::types::*;

pub type Pivot = Vec<i32>;

pub trait LapackScalar
    : OperatorNorm_ + QR_ + SVD_ + Solve_ + Cholesky_ + Eigh_ + Triangular_ {
}

impl LapackScalar for f32 {}
impl LapackScalar for f64 {}
impl LapackScalar for c32 {}
impl LapackScalar for c64 {}

pub fn into_result<T>(info: i32, val: T) -> Result<T> {
    if info == 0 {
        Ok(val)
    } else {
        Err(LapackError::new(info).into())
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
