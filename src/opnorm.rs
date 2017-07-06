//! Operator norm

use ndarray::*;

use super::error::*;
use super::layout::*;
use super::types::*;

pub use lapack_traits::NormType;

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
where
    A: Scalar,
    S: Data<Elem = A>,
{
    type Output = Result<A::Real>;

    fn opnorm(&self, t: NormType) -> Self::Output {
        let l = self.layout()?;
        let a = self.as_allocated()?;
        Ok(A::opnorm(t, l, a))
    }
}
