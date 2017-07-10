//! QR decomposition
//!
//! https://en.wikipedia.org/wiki/QR_decomposition

use ndarray::*;
use num_traits::Zero;

use super::convert::*;
use super::error::*;
use super::layout::*;
use super::triangular::*;
use super::types::*;

pub use lapack_traits::UPLO;

/// QR decomposition for matrix reference
///
/// This creates copy due for reshaping array.
/// To avoid copy and the matrix is square, please use `QRSquare*` traits.
pub trait QR {
    type Q;
    type R;
    fn qr(&self) -> Result<(Self::Q, Self::R)>;
}

/// QR decomposition
///
/// This creates copy due for reshaping array.
/// To avoid copy and the matrix is square, please use `QRSquare*` traits.
pub trait QRInto: Sized {
    type Q;
    type R;
    fn qr_into(self) -> Result<(Self::Q, Self::R)>;
}

/// QR decomposition for square matrix reference
pub trait QRSquare: Sized {
    type Q;
    type R;
    fn qr_square(&self) -> Result<(Self::Q, Self::R)>;
}

/// QR decomposition for square matrix
pub trait QRSquareInto: Sized {
    type R;
    fn qr_square_into(self) -> Result<(Self, Self::R)>;
}

/// QR decomposition for mutable reference of square matrix
pub trait QRSquareMut: Sized {
    type R;
    fn qr_square_mut<'a>(&'a mut self) -> Result<(&'a mut Self, Self::R)>;
}

impl<A, S> QRSquareMut for ArrayBase<S, Ix2>
where
    A: Scalar,
    S: DataMut<Elem = A>,
{
    type R = Array2<A>;

    fn qr_square_mut<'a>(&'a mut self) -> Result<(&'a mut Self, Self::R)> {
        let l = self.square_layout()?;
        let r = unsafe { A::qr(l, self.as_allocated_mut()?)? };
        let r: Array2<_> = into_matrix(l, r)?;
        Ok((self, r.into_triangular(UPLO::Upper)))
    }
}

impl<A, S> QRSquareInto for ArrayBase<S, Ix2>
where
    A: Scalar,
    S: DataMut<Elem = A>,
{
    type R = Array2<A>;

    fn qr_square_into(mut self) -> Result<(Self, Self::R)> {
        let (_, r) = self.qr_square_mut()?;
        Ok((self, r))
    }
}

impl<A, S> QRSquare for ArrayBase<S, Ix2>
where
    A: Scalar,
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
    A: Scalar,
    S: DataMut<Elem = A>,
{
    type Q = Array2<A>;
    type R = Array2<A>;

    fn qr_into(mut self) -> Result<(Self::Q, Self::R)> {
        let n = self.rows();
        let m = self.cols();
        let k = ::std::cmp::min(n, m);
        let l = self.layout()?;
        let r = unsafe { A::qr(l, self.as_allocated_mut()?)? };
        let r: Array2<_> = into_matrix(l, r)?;
        let q = self;
        Ok((take_slice(&q, n, k), take_slice_upper(&r, k, m)))
    }
}

impl<A, S> QR for ArrayBase<S, Ix2>
where
    A: Scalar,
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
