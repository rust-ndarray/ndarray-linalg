//! Solve linear problems

use ndarray::*;

use super::convert::*;
use super::error::*;
use super::layout::*;
use super::types::*;

pub use lapack_traits::{Pivot, Transpose};

pub struct Factorized<S: Data> {
    pub a: ArrayBase<S, Ix2>,
    pub ipiv: Pivot,
}

impl<A, S> Factorized<S>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    pub fn solve<Sb>(&self, t: Transpose, mut rhs: ArrayBase<Sb, Ix1>) -> Result<ArrayBase<Sb, Ix1>>
    where
        Sb: DataMut<Elem = A>,
    {
        A::solve(
            self.a.square_layout()?,
            t,
            self.a.as_allocated()?,
            &self.ipiv,
            rhs.as_slice_mut().unwrap(),
        )?;
        Ok(rhs)
    }
}

impl<A, S> Factorized<S>
where
    A: Scalar,
    S: DataMut<Elem = A>,
{
    pub fn into_inverse(mut self) -> Result<ArrayBase<S, Ix2>> {
        A::inv(
            self.a.square_layout()?,
            self.a.as_allocated_mut()?,
            &self.ipiv,
        )?;
        Ok(self.a)
    }
}

pub trait Factorize<S: Data> {
    fn factorize(&self) -> Result<Factorized<S>>;
}

pub trait FactorizeInto<S: Data> {
    fn factorize_into(self) -> Result<Factorized<S>>;
}

impl<A, S> FactorizeInto<S> for ArrayBase<S, Ix2>
where
    A: Scalar,
    S: DataMut<Elem = A>,
{
    fn factorize_into(mut self) -> Result<Factorized<S>> {
        let ipiv = A::lu(self.layout()?, self.as_allocated_mut()?)?;
        Ok(Factorized {
            a: self,
            ipiv: ipiv,
        })
    }
}

impl<A, Si, So> Factorize<So> for ArrayBase<Si, Ix2>
where
    A: Scalar,
    Si: Data<Elem = A>,
    So: DataOwned<Elem = A> + DataMut,
{
    fn factorize(&self) -> Result<Factorized<So>> {
        let mut a: ArrayBase<So, Ix2> = replicate(self);
        let ipiv = A::lu(a.layout()?, a.as_allocated_mut()?)?;
        Ok(Factorized { a: a, ipiv: ipiv })
    }
}

pub trait Inverse {
    type Output;
    fn inv(&self) -> Result<Self::Output>;
}

pub trait InverseInto {
    type Output;
    fn inv_into(self) -> Result<Self::Output>;
}

impl<A, S> InverseInto for ArrayBase<S, Ix2>
where
    A: Scalar,
    S: DataMut<Elem = A>,
{
    type Output = Self;

    fn inv_into(self) -> Result<Self::Output> {
        let f = self.factorize_into()?;
        f.into_inverse()
    }
}

impl<A, Si> Inverse for ArrayBase<Si, Ix2>
where
    A: Scalar,
    Si: Data<Elem = A>,
{
    type Output = Array2<A>;

    fn inv(&self) -> Result<Self::Output> {
        let f = self.factorize()?;
        f.into_inverse()
    }
}
