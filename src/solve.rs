
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
        A::solve(self.a.layout()?,
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
        A::inv(self.a.layout()?, self.a.as_allocated_mut()?, &self.ipiv)?;
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

impl<'a, A, S> Factorize<OwnedRepr<A>> for &'a ArrayBase<S, Ix2>
    where A: LapackScalar + Clone,
          S: Data<Elem = A>
{
    fn factorize(self) -> Result<Factorized<OwnedRepr<A>>> {
        let mut a = self.to_owned();
        let ipiv = A::lu(a.layout()?, a.as_allocated_mut()?)?;
        Ok(Factorized { a: a, ipiv: ipiv })
    }
}
