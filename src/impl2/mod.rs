
pub mod opnorm;
pub mod qr;
pub use self::opnorm::*;
pub use self::qr::*;

pub trait LapackScalar: OperatorNorm_ + QR_ {}
impl<A> LapackScalar for A where A: OperatorNorm_ + QR_ {}
