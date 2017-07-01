//! QR decomposition

use ndarray::*;
use num_traits::Zero;

use super::convert::*;
use super::error::*;
use super::layout::*;
use super::triangular::*;

use lapack_traits::{LapackScalar, UPLO};

pub trait QR {
    type Q;
    type R;
    fn qr(&self) -> Result<(Self::Q, Self::R)>;
}

pub trait QRInto: Sized {
    type Q;
    type R;
    fn qr_into(self) -> Result<(Self::Q, Self::R)>;
}

pub trait QRSquare: Sized {
    type Q;
    type R;
    fn qr_square(&self) -> Result<(Self::Q, Self::R)>;
}

pub trait QRSquareInto: Sized {
    type R;
    fn qr_square_into(self) -> Result<(Self, Self::R)>;
}

impl<A, S> QRSquareInto for ArrayBase<S, Ix2>
where
    A: LapackScalar + Copy + Zero,
    S: DataMut<Elem = A>,
{
    type R = Array2<A>;

    fn qr_square_into(mut self) -> Result<(Self, Self::R)> {
        let l = self.square_layout()?;
        let r = A::qr(l, self.as_allocated_mut()?)?;
        let r: Array2<_> = into_matrix(l, r)?;
        Ok((self, r.into_triangular(UPLO::Upper)))
    }
}

impl<A, S> QRSquare for ArrayBase<S, Ix2>
where
    A: LapackScalar + Copy + Zero,
    S: Data<Elem = A>,
{
    type Q = Array2<A>;
    type R = Array2<A>;

    fn qr_square(&self) -> Result<(Self::Q, Self::R)> {
        let a = self.to_owned();
        a.qr_square_into()
    }
}


impl<A, S> QRInto for ArrayBase<S, Ix2>
where
    A: LapackScalar + Copy + Zero,
    S: DataMut<Elem = A>,
{
    type Q = Array2<A>;
    type R = Array2<A>;

    fn qr_into(mut self) -> Result<(Self::Q, Self::R)> {
        let n = self.rows();
        let m = self.cols();
        let k = ::std::cmp::min(n, m);
        let l = self.layout()?;
        let r = A::qr(l, self.as_allocated_mut()?)?;
        let r: Array2<_> = into_matrix(l, r)?;
        let q = self;
        Ok((take_slice(&q, n, k), take_slice_upper(&r, k, m)))
    }
}

impl<A, S> QR for ArrayBase<S, Ix2>
where
    A: LapackScalar + Copy + Zero,
    S: Data<Elem = A>,
{
    type Q = Array2<A>;
    type R = Array2<A>;

    fn qr(&self) -> Result<(Self::Q, Self::R)> {
        let a = self.to_owned();
        a.qr_into()
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
