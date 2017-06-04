
pub use impl2::LapackScalar;
pub use impl2::NormType;

use num_traits::Zero;
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
    fn qr(self) -> Result<(Q, R)>;
}

impl<A, S, Sq, Sr> QR<ArrayBase<Sq, Ix2>, ArrayBase<Sr, Ix2>> for ArrayBase<S, Ix2>
    where A: LapackScalar + Copy + Zero,
          S: DataMut<Elem = A>,
          Sq: DataOwned<Elem = A> + DataMut,
          Sr: DataOwned<Elem = A> + DataMut
{
    fn qr(mut self) -> Result<(ArrayBase<Sq, Ix2>, ArrayBase<Sr, Ix2>)> {
        let n = self.rows();
        let m = self.cols();
        let k = ::std::cmp::min(n, m);
        let l = self.layout()?;
        // calc QR decomposition
        let r = A::qr(l, self.as_allocated_mut()?)?;
        let r: Array2<_> = reconstruct(l, r)?;
        let q = self;
        // get slice
        let qv = q.slice(s![..n as isize, ..k as isize]);
        let mut q = unsafe { ArrayBase::uninitialized((n, k)) };
        q.assign(&qv);
        let rv = r.slice(s![..k as isize, ..m as isize]);
        let mut r = ArrayBase::zeros((k, m));
        for ((i, j), val) in r.indexed_iter_mut() {
            if i <= j {
                *val = rv[(i, j)];
            }
        }
        Ok((q, r))
    }
}
