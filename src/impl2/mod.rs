
pub mod opnorm;
pub mod qr;
pub mod svd;

pub use self::opnorm::*;
pub use self::qr::*;
pub use self::svd::*;

pub trait LapackScalar: OperatorNorm_ + QR_ + SVD_ {}
impl<A> LapackScalar for A where A: OperatorNorm_ + QR_ + SVD_ {}
