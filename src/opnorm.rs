//! Operator norm

use ndarray::*;

use crate::convert::*;
use crate::error::*;
use crate::layout::*;
use crate::tridiagonal::TriDiagonal;
use crate::types::*;

pub use crate::lapack::NormType;

/// Operator norm using `*lange` LAPACK routines
///
/// [Wikipedia article on operator norm](https://en.wikipedia.org/wiki/Operator_norm)
pub trait OperationNorm {
    /// the value of norm
    type Output: Scalar;

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
    A: Scalar + Lapack,
    S: Data<Elem = A>,
{
    type Output = A::Real;

    fn opnorm(&self, t: NormType) -> Result<Self::Output> {
        let l = self.layout()?;
        let a = self.as_allocated()?;
        Ok(unsafe { A::opnorm(t, l, a) })
    }
}

impl<A> OperationNorm for TriDiagonal<A>
where
    A: Scalar + Lapack,
{
    type Output = A::Real;

    fn opnorm(&self, t: NormType) -> Result<Self::Output> {
        let arr = match t {
            NormType::One => {
                let zl: Array1<A> = Array::zeros(1);
                let zu: Array1<A> = Array::zeros(1);
                let dl = stack![Axis(0), self.dl.to_owned(), zl];
                let du = stack![Axis(0), zu, self.du.to_owned()];
                let arr = stack![
                    Axis(0),
                    into_row(du),
                    into_row(self.d.to_owned()),
                    into_row(dl)
                ];
                arr
            }
            NormType::Infinity => {
                let zl: Array1<A> = Array::zeros(1);
                let zu: Array1<A> = Array::zeros(1);
                let dl = stack![Axis(0), zl, self.dl.to_owned()];
                let du = stack![Axis(0), self.du.to_owned(), zu];
                let arr = stack![
                    Axis(1),
                    into_col(dl),
                    into_col(self.d.to_owned()),
                    into_col(du)
                ];
                arr
            }
            NormType::Frobenius => {
                let arr = stack![
                    Axis(1),
                    into_row(self.dl.to_owned()),
                    into_row(self.d.to_owned()),
                    into_row(self.du.to_owned())
                ];
                arr
            }
        };

        let l = arr.layout()?;
        let a = arr.as_allocated()?;
        Ok(unsafe { A::opnorm(t, l, a) })
    }
}
