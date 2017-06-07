
pub mod opnorm;
pub mod qr;
pub mod svd;

pub use self::opnorm::*;
pub use self::qr::*;
pub use self::svd::*;

use super::error::*;

pub trait LapackScalar: OperatorNorm_ + QR_ + SVD_ {}
impl<A> LapackScalar for A where A: OperatorNorm_ + QR_ + SVD_ {}

pub fn into_result<T>(info: i32, val: T) -> Result<T> {
    if info == 0 {
        Ok(val)
    } else {
        Err(LapackError::new(info).into())
    }
}
