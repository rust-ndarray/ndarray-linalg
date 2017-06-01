
pub mod opnorm;
pub use self::opnorm::*;

pub trait LapackScalar: OperatorNorm_ {}
impl<A> LapackScalar for A where A: OperatorNorm_ {}
