//! Operator norm

use lax::Tridiagonal;
use ndarray::*;

use crate::error::*;
use crate::layout::*;
use crate::types::*;

pub use lax::NormType;

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
        Ok(A::opnorm(t, l, a))
    }
}

impl<A> OperationNorm for Tridiagonal<A>
where
    A: Scalar + Lapack,
{
    type Output = A::Real;

    fn opnorm(&self, t: NormType) -> Result<Self::Output> {
        // `self` is a tridiagonal matrix like,
        // [d0, u1,  0,   ...,       0,
        //  l1, d1, u2,            ...,
        //   0, l2, d2,
        //  ...           ...,  u{n-1},
        //   0,  ...,  l{n-1},  d{n-1},]
        let arr = match t {
            // opnorm_one() calculates muximum column sum.
            // Therefore, This part align the columns and make a (3 x n) matrix like,
            // [ 0, u1, u2, ..., u{n-1},
            //  d0, d1, d2, ..., d{n-1},
            //  l1, l2, l3, ...,      0,]
            NormType::One => {
                let zl: Array1<A> = Array::zeros(1);
                let zu: Array1<A> = Array::zeros(1);
                let dl = concatenate![Axis(0), &self.dl, zl]; // n
                let du = concatenate![Axis(0), zu, &self.du]; // n
                stack![Axis(0), du, &self.d, dl] // 3 x n
            }
            // opnorm_inf() calculates muximum row sum.
            // Therefore, This part align the rows and make a (n x 3) matrix like,
            // [     0,     d0,     u1,
            //      l1,     d1,     u2,
            //      l2,     d2,     u3,
            //     ...,    ...,    ...,
            //  l{n-1}, d{n-1},      0,]
            NormType::Infinity => {
                let zl: Array1<A> = Array::zeros(1);
                let zu: Array1<A> = Array::zeros(1);
                let dl = concatenate![Axis(0), zl, &self.dl]; // n
                let du = concatenate![Axis(0), &self.du, zu]; // n
                stack![Axis(1), dl, &self.d, du] // n x 3
            }
            // opnorm_fro() calculates square root of sum of squares.
            // Because it is independent of the shape of matrix,
            // this part make a (1 x (3n-2)) matrix like,
            // [l1, ..., l{n-1}, d0, ..., d{n-1}, u1, ..., u{n-1}]
            NormType::Frobenius => {
                concatenate![Axis(0), &self.dl, &self.d, &self.du].insert_axis(Axis(0))
            }
        };
        let l = arr.layout()?;
        let a = arr.as_allocated()?;
        Ok(A::opnorm(t, l, a))
    }
}
