
use ndarray::*;
use super::layout::*;
use super::error::*;
use super::impl2::*;

pub use impl2::{Pivot, Transpose};

pub struct Factorized<S: Data> {
    pub a: ArrayBase<S, Ix2>,
    pub ipiv: Pivot,
}

impl<A, S> Factorized<S>
    where A: LapackScalar,
          S: Data<Elem = A>
{
    pub fn solve<Sb>(&self, t: Transpose, mut rhs: ArrayBase<Sb, Ix1>) -> Result<ArrayBase<Sb, Ix1>>
        where Sb: DataMut<Elem = A>
    {
        A::solve(self.a.square_layout()?,
                 t,
                 self.a.as_allocated()?,
                 &self.ipiv,
                 rhs.as_slice_mut().unwrap())?;
        Ok(rhs)
    }
}

impl<A, S> Factorized<S>
    where A: LapackScalar,
          S: DataMut<Elem = A>
{
    pub fn into_inverse(mut self) -> Result<ArrayBase<S, Ix2>> {
        A::inv(self.a.square_layout()?,
               self.a.as_allocated_mut()?,
               &self.ipiv)?;
        Ok(self.a)
    }
}

pub trait Factorize<S: Data> {
    fn factorize(self) -> Result<Factorized<S>>;
}

impl<A, S> Factorize<S> for ArrayBase<S, Ix2>
    where A: LapackScalar,
          S: DataMut<Elem = A>
{
    fn factorize(mut self) -> Result<Factorized<S>> {
        let ipiv = A::lu(self.layout()?, self.as_allocated_mut()?)?;
        Ok(Factorized {
            a: self,
            ipiv: ipiv,
        })
    }
}

impl<'a, A, Si, So> Factorize<So> for &'a ArrayBase<Si, Ix2>
    where A: LapackScalar + Copy,
          Si: Data<Elem = A>,
          So: DataOwned<Elem = A> + DataMut
{
    fn factorize(self) -> Result<Factorized<So>> {
        let mut a: ArrayBase<So, Ix2> = replicate(self);
        let ipiv = A::lu(a.layout()?, a.as_allocated_mut()?)?;
        Ok(Factorized { a: a, ipiv: ipiv })
    }
}

pub trait Inverse<Inv> {
    fn inv(self) -> Result<Inv>;
}

impl<A, S> Inverse<ArrayBase<S, Ix2>> for ArrayBase<S, Ix2>
    where A: LapackScalar,
          S: DataMut<Elem = A>
{
    fn inv(self) -> Result<ArrayBase<S, Ix2>> {
        let f = self.factorize()?;
        f.into_inverse()
    }
}

impl<'a, A, Si, So> Inverse<ArrayBase<So, Ix2>> for &'a ArrayBase<Si, Ix2>
    where A: LapackScalar + Copy,
          Si: Data<Elem = A>,
          So: DataOwned<Elem = A> + DataMut
{
    fn inv(self) -> Result<ArrayBase<So, Ix2>> {
        let f = self.factorize()?;
        f.into_inverse()
    }
}
