
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

pub trait QR<Q, R> {
    fn qr2(self) -> Result<(Q, R)>;
}

impl<A, Sq, Sr> QR<ArrayBase<Sq, Ix2>, ArrayBase<Sr, Ix2>> for ArrayBase<Sq, Ix2>
    where A: LapackScalar,
          Sq: DataMut<Elem = A>,
          Sr: DataOwned<Elem = A>
{
    fn qr2(mut self) -> Result<(ArrayBase<Sq, Ix2>, ArrayBase<Sr, Ix2>)> {
        let l = self.layout()?;
        let r = A::qr(l, self.as_allocated_mut()?)?;
        let r = reconstruct(l, r)?;
        let q = self;
        Ok((q, r))
    }
}
