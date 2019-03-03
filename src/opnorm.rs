//! Operator norm

use ndarray::*;

use crate::error::*;
use crate::layout::*;
use crate::types::*;

pub use crate::lapack_traits::NormType;

/// Operator norm using `*lange` LAPACK routines
///
/// [Wikipedia article on operator norm](https://en.wikipedia.org/wiki/Operator_norm)
pub trait OperationNorm {
    /// the value of norm
    type Output: RealScalar;

    fn opnorm(&self, t: NormType) -> Result<Self::Output>;

    /// the one norm of a matrix (maximum column sum)
    fn opnorm_one(&self) -> Result<Self::Output> {
        self.opnorm(NormType::One)
    }

    /// the infinity norm of a matrix (maximum row sum)
    fn opnorm_inf(&self) -> Result<Self::Output> {
        self.opnorm(NormType::Infinity)
    }

    /// the Frobenius norm of a matrix (square root of sum of squares)
    fn opnorm_fro(&self) -> Result<Self::Output> {
        self.opnorm(NormType::Frobenius)
    }
}

impl<A, S> OperationNorm for ArrayBase<S, Ix2>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    type Output = A::Real;

    fn opnorm(&self, t: NormType) -> Result<Self::Output> {
        let l = self.layout()?;
        let a = self.as_allocated()?;
        Ok(unsafe { A::opnorm(t, l, a) })
    }
}
