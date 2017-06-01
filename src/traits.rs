
pub use impl2::LapackScalar;
pub use impl2::NormType;

use ndarray::*;

use super::types::*;
use super::error::*;
use super::layout::*;

pub trait OperationNorm {
    type Output;
    fn opnorm(&self, t: NormType) -> Self::Output;
    fn opnorm_one(&self) -> Self::Output {
        self.opnorm(NormType::One)
    }
    fn opnorm_inf(&self) -> Self::Output {
        self.opnorm(NormType::Infinity)
    }
    fn opnorm_fro(&self) -> Self::Output {
        self.opnorm(NormType::Frobenius)
    }
}

impl<A, S> OperationNorm for ArrayBase<S, Ix2>
    where A: LapackScalar + AssociatedReal,
          S: Data<Elem = A>
{
    type Output = Result<A::Real>;

    fn opnorm(&self, t: NormType) -> Self::Output {
        let l = self.layout()?;
        let a = self.as_allocated()?;
        Ok(A::opnorm(t, l, a))
    }
}
