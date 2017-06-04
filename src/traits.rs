
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
        (&mut self).qr()
    }
}

fn take_slice<A, S1, S2>(a: &ArrayBase<S1, Ix2>, n: usize, m: usize) -> ArrayBase<S2, Ix2>
    where A: Copy,
          S1: Data<Elem = A>,
          S2: DataMut<Elem = A> + DataOwned
{
    let av = a.slice(s![..n as isize, ..m as isize]);
    let mut a = unsafe { ArrayBase::uninitialized((n, m)) };
    a.assign(&av);
    a
}

fn take_slice_upper<A, S1, S2>(a: &ArrayBase<S1, Ix2>, n: usize, m: usize) -> ArrayBase<S2, Ix2>
    where A: Copy + Zero,
          S1: Data<Elem = A>,
          S2: DataMut<Elem = A> + DataOwned
{
    let av = a.slice(s![..n as isize, ..m as isize]);
    let mut a = unsafe { ArrayBase::uninitialized((n, m)) };
    for ((i, j), val) in a.indexed_iter_mut() {
        *val = if i <= j { av[(i, j)] } else { A::zero() };
    }
    a
}

impl<'a, A, S, Sq, Sr> QR<ArrayBase<Sq, Ix2>, ArrayBase<Sr, Ix2>> for &'a mut ArrayBase<S, Ix2>
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
        let r = A::qr(l, self.as_allocated_mut()?)?;
        let r: Array2<_> = reconstruct(l, r)?;
        let q = self;
        Ok((take_slice(q, n, k), take_slice_upper(&r, k, m)))
    }
}

impl<'a, A, S, Sq, Sr> QR<ArrayBase<Sq, Ix2>, ArrayBase<Sr, Ix2>> for &'a ArrayBase<S, Ix2>
    where A: LapackScalar + Copy + Zero,
          S: Data<Elem = A>,
          Sq: DataOwned<Elem = A> + DataMut,
          Sr: DataOwned<Elem = A> + DataMut
{
    fn qr(self) -> Result<(ArrayBase<Sq, Ix2>, ArrayBase<Sr, Ix2>)> {
        let n = self.rows();
        let m = self.cols();
        let k = ::std::cmp::min(n, m);
        let l = self.layout()?;
        let mut q = self.to_owned();
        let r = A::qr(l, q.as_allocated_mut()?)?;
        let r: Array2<_> = reconstruct(l, r)?;
        Ok((take_slice(&q, n, k), take_slice_upper(&r, k, m)))
    }
}
