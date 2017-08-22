//! Solve linear problems

use ndarray::*;

use super::convert::*;
use super::error::*;
use super::layout::*;
use super::types::*;

pub use lapack_traits::{Pivot, Transpose};

pub trait Solve<A: Scalar> {
    fn solve<S: Data<Elem = A>>(&self, a: &ArrayBase<S, Ix1>) -> Result<Array1<A>> {
        let mut a = replicate(a);
        self.solve_mut(&mut a)?;
        Ok(a)
    }
    fn solve_into<S: DataMut<Elem = A>>(&self, mut a: ArrayBase<S, Ix1>) -> Result<ArrayBase<S, Ix1>> {
        self.solve_mut(&mut a)?;
        Ok(a)
    }
    fn solve_mut<'a, S: DataMut<Elem = A>>(&self, &'a mut ArrayBase<S, Ix1>) -> Result<&'a mut ArrayBase<S, Ix1>>;

    fn solve_t<S: Data<Elem = A>>(&self, a: &ArrayBase<S, Ix1>) -> Result<Array1<A>> {
        let mut a = replicate(a);
        self.solve_t_mut(&mut a)?;
        Ok(a)
    }
    fn solve_t_into<S: DataMut<Elem = A>>(&self, mut a: ArrayBase<S, Ix1>) -> Result<ArrayBase<S, Ix1>> {
        self.solve_t_mut(&mut a)?;
        Ok(a)
    }
    fn solve_t_mut<'a, S: DataMut<Elem = A>>(&self, &'a mut ArrayBase<S, Ix1>) -> Result<&'a mut ArrayBase<S, Ix1>>;

    fn solve_h<S: Data<Elem = A>>(&self, a: &ArrayBase<S, Ix1>) -> Result<Array1<A>> {
        let mut a = replicate(a);
        self.solve_h_mut(&mut a)?;
        Ok(a)
    }
    fn solve_h_into<S: DataMut<Elem = A>>(&self, mut a: ArrayBase<S, Ix1>) -> Result<ArrayBase<S, Ix1>> {
        self.solve_h_mut(&mut a)?;
        Ok(a)
    }
    fn solve_h_mut<'a, S: DataMut<Elem = A>>(&self, &'a mut ArrayBase<S, Ix1>) -> Result<&'a mut ArrayBase<S, Ix1>>;
}

pub struct Factorized<S: Data> {
    pub a: ArrayBase<S, Ix2>,
    pub ipiv: Pivot,
}

impl<A, S> Solve<A> for Factorized<S>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    fn solve_mut<'a, Sb>(&self, mut rhs: &'a mut ArrayBase<Sb, Ix1>) -> Result<&'a mut ArrayBase<Sb, Ix1>>
    where
        Sb: DataMut<Elem = A>,
    {
        unsafe {
            A::solve(
                self.a.square_layout()?,
                Transpose::No,
                self.a.as_allocated()?,
                &self.ipiv,
                rhs.as_slice_mut().unwrap(),
            )?
        };
        Ok(rhs)
    }
    fn solve_t_mut<'a, Sb>(&self, mut rhs: &'a mut ArrayBase<Sb, Ix1>) -> Result<&'a mut ArrayBase<Sb, Ix1>>
    where
        Sb: DataMut<Elem = A>,
    {
        unsafe {
            A::solve(
                self.a.square_layout()?,
                Transpose::Transpose,
                self.a.as_allocated()?,
                &self.ipiv,
                rhs.as_slice_mut().unwrap(),
            )?
        };
        Ok(rhs)
    }
    fn solve_h_mut<'a, Sb>(&self, mut rhs: &'a mut ArrayBase<Sb, Ix1>) -> Result<&'a mut ArrayBase<Sb, Ix1>>
    where
        Sb: DataMut<Elem = A>,
    {
        unsafe {
            A::solve(
                self.a.square_layout()?,
                Transpose::Hermite,
                self.a.as_allocated()?,
                &self.ipiv,
                rhs.as_slice_mut().unwrap(),
            )?
        };
        Ok(rhs)
    }
}

impl<A, S> Solve<A> for ArrayBase<S, Ix2>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    fn solve_mut<'a, Sb>(&self, mut rhs: &'a mut ArrayBase<Sb, Ix1>) -> Result<&'a mut ArrayBase<Sb, Ix1>>
    where
        Sb: DataMut<Elem = A>,
    {
        let f = self.factorize()?;
        f.solve_mut(rhs)
    }
    fn solve_t_mut<'a, Sb>(&self, mut rhs: &'a mut ArrayBase<Sb, Ix1>) -> Result<&'a mut ArrayBase<Sb, Ix1>>
    where
        Sb: DataMut<Elem = A>,
    {
        let f = self.factorize()?;
        f.solve_t_mut(rhs)
    }
    fn solve_h_mut<'a, Sb>(&self, mut rhs: &'a mut ArrayBase<Sb, Ix1>) -> Result<&'a mut ArrayBase<Sb, Ix1>>
    where
        Sb: DataMut<Elem = A>,
    {
        let f = self.factorize()?;
        f.solve_h_mut(rhs)
    }
}

impl<A, S> Factorized<S>
where
    A: Scalar,
    S: DataMut<Elem = A>,
{
    pub fn into_inverse(mut self) -> Result<ArrayBase<S, Ix2>> {
        unsafe {
            A::inv(
                self.a.square_layout()?,
                self.a.as_allocated_mut()?,
                &self.ipiv,
            )?
        };
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
        let ipiv = unsafe { A::lu(self.layout()?, self.as_allocated_mut()?)? };
        Ok(Factorized {
            a: self,
            ipiv: ipiv,
        })
    }
}

impl<A, Si> Factorize<OwnedRepr<A>> for ArrayBase<Si, Ix2>
where
    A: Scalar,
    Si: Data<Elem = A>,
{
    fn factorize(&self) -> Result<Factorized<OwnedRepr<A>>> {
        let mut a: Array2<A> = replicate(self);
        let ipiv = unsafe { A::lu(a.layout()?, a.as_allocated_mut()?)? };
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
