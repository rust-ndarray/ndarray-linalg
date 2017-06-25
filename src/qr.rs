//! QR decomposition

use ndarray::*;
use num_traits::Zero;

use super::error::*;
use super::layout::*;

use lapack_traits::LapackScalar;

pub trait QR<Q, R> {
    fn qr(self) -> Result<(Q, R)>;
}

impl<A, S, Sq, Sr> QR<ArrayBase<Sq, Ix2>, ArrayBase<Sr, Ix2>> for ArrayBase<S, Ix2>
where
    A: LapackScalar + Copy + Zero,
    S: DataMut<Elem = A>,
    Sq: DataOwned<Elem = A> + DataMut,
    Sr: DataOwned<Elem = A> + DataMut,
{
    fn qr(mut self) -> Result<(ArrayBase<Sq, Ix2>, ArrayBase<Sr, Ix2>)> {
        (&mut self).qr()
    }
}

fn take_slice<A, S1, S2>(a: &ArrayBase<S1, Ix2>, n: usize, m: usize) -> ArrayBase<S2, Ix2>
where
    A: Copy,
    S1: Data<Elem = A>,
    S2: DataMut<Elem = A> + DataOwned,
{
    let av = a.slice(s![..n as isize, ..m as isize]);
    let mut a = unsafe { ArrayBase::uninitialized((n, m)) };
    a.assign(&av);
    a
}

fn take_slice_upper<A, S1, S2>(a: &ArrayBase<S1, Ix2>, n: usize, m: usize) -> ArrayBase<S2, Ix2>
where
    A: Copy + Zero,
    S1: Data<Elem = A>,
    S2: DataMut<Elem = A> + DataOwned,
{
    let av = a.slice(s![..n as isize, ..m as isize]);
    let mut a = unsafe { ArrayBase::uninitialized((n, m)) };
    for ((i, j), val) in a.indexed_iter_mut() {
        *val = if i <= j { av[(i, j)] } else { A::zero() };
    }
    a
}

impl<'a, A, S, Sq, Sr> QR<ArrayBase<Sq, Ix2>, ArrayBase<Sr, Ix2>> for &'a mut ArrayBase<S, Ix2>
where
    A: LapackScalar
        + Copy
        + Zero,
    S: DataMut<Elem = A>,
    Sq: DataOwned<Elem = A>
        + DataMut,
    Sr: DataOwned<Elem = A>
        + DataMut,
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
