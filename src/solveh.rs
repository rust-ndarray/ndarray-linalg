//! Solve Hermite/Symmetric linear problems

use ndarray::*;

use super::convert::*;
use super::error::*;
use super::layout::*;
use super::types::*;

pub use lapack_traits::{Pivot, UPLO};

pub trait SolveH<A: Scalar> {
    fn solveh<S: Data<Elem = A>>(&self, a: &ArrayBase<S, Ix1>) -> Result<Array1<A>> {
        let mut a = replicate(a);
        self.solveh_mut(&mut a)?;
        Ok(a)
    }
    fn solveh_into<S: DataMut<Elem = A>>(&self, mut a: ArrayBase<S, Ix1>) -> Result<ArrayBase<S, Ix1>> {
        self.solveh_mut(&mut a)?;
        Ok(a)
    }
    fn solveh_mut<'a, S: DataMut<Elem = A>>(&self, &'a mut ArrayBase<S, Ix1>) -> Result<&'a mut ArrayBase<S, Ix1>>;
}

pub struct FactorizedH<S: Data> {
    pub a: ArrayBase<S, Ix2>,
    pub ipiv: Pivot,
}

impl<A, S> SolveH<A> for FactorizedH<S>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    fn solveh_mut<'a, Sb>(&self, rhs: &'a mut ArrayBase<Sb, Ix1>) -> Result<&'a mut ArrayBase<Sb, Ix1>>
    where
        Sb: DataMut<Elem = A>,
    {
        unsafe {
            A::solveh(
                self.a.square_layout()?,
                UPLO::Upper,
                self.a.as_allocated()?,
                &self.ipiv,
                rhs.as_slice_mut().unwrap(),
            )?
        };
        Ok(rhs)
    }
}

impl<A, S> FactorizedH<S>
where
    A: Scalar,
    S: DataMut<Elem = A>,
{
    pub fn into_inverseh(mut self) -> Result<ArrayBase<S, Ix2>> {
        unsafe {
            A::invh(
                self.a.square_layout()?,
                UPLO::Upper,
                self.a.as_allocated_mut()?,
                &self.ipiv,
            )?
        };
        Ok(self.a)
    }
}

pub trait FactorizeH<S: Data> {
    fn factorizeh(&self) -> Result<FactorizedH<S>>;
}

pub trait FactorizeHInto<S: Data> {
    fn factorizeh_into(self) -> Result<FactorizedH<S>>;
}

impl<A, S> FactorizeHInto<S> for ArrayBase<S, Ix2>
where
    A: Scalar,
    S: DataMut<Elem = A>,
{
    fn factorizeh_into(mut self) -> Result<FactorizedH<S>> {
        let ipiv = unsafe { A::bk(self.layout()?, UPLO::Upper, self.as_allocated_mut()?)? };
        Ok(FactorizedH {
            a: self,
            ipiv: ipiv,
        })
    }
}

impl<A, Si> FactorizeH<OwnedRepr<A>> for ArrayBase<Si, Ix2>
where
    A: Scalar,
    Si: Data<Elem = A>,
{
    fn factorizeh(&self) -> Result<FactorizedH<OwnedRepr<A>>> {
        let mut a: Array2<A> = replicate(self);
        let ipiv = unsafe { A::bk(a.layout()?, UPLO::Upper, a.as_allocated_mut()?)? };
        Ok(FactorizedH { a: a, ipiv: ipiv })
    }
}

pub trait InverseH {
    type Output;
    fn invh(&self) -> Result<Self::Output>;
}

pub trait InverseHInto {
    type Output;
    fn invh_into(self) -> Result<Self::Output>;
}

impl<A, S> InverseHInto for ArrayBase<S, Ix2>
where
    A: Scalar,
    S: DataMut<Elem = A>,
{
    type Output = Self;

    fn invh_into(self) -> Result<Self::Output> {
        let f = self.factorizeh_into()?;
        f.into_inverseh()
    }
}

impl<A, Si> InverseH for ArrayBase<Si, Ix2>
where
    A: Scalar,
    Si: Data<Elem = A>,
{
    type Output = Array2<A>;

    fn invh(&self) -> Result<Self::Output> {
        let f = self.factorizeh()?;
        f.into_inverseh()
    }
}
