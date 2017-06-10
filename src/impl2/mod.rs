
pub mod opnorm;
pub mod qr;
pub mod svd;
pub mod solve;
pub mod cholesky;

pub use self::opnorm::*;
pub use self::qr::*;
pub use self::svd::*;
pub use self::solve::*;
pub use self::cholesky::*;

use super::error::*;

pub trait LapackScalar: OperatorNorm_ + QR_ + SVD_ + Solve_ + Cholesky_ {}
impl<A> LapackScalar for A where A: OperatorNorm_ + QR_ + SVD_ + Solve_ + Cholesky_ {}

pub fn into_result<T>(info: i32, val: T) -> Result<T> {
    if info == 0 {
        Ok(val)
    } else {
        Err(LapackError::new(info).into())
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum UPLO {
    Upper = b'U',
    Lower = b'L',
}
