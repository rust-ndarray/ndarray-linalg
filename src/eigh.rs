//! Eigenvalue decomposition for Hermite matrices

use ndarray::*;
use num_traits::Float;

use super::convert::*;
use super::diagonal::*;
use super::error::*;
use super::layout::*;
use super::operator::*;
use super::types::*;

use lapack_traits::LapackScalar;
pub use lapack_traits::UPLO;

pub trait Eigh<EigVal, EigVec> {
    fn eigh(self, UPLO) -> Result<(EigVal, EigVec)>;
}

impl<A, S, Se> Eigh<ArrayBase<Se, Ix1>, ArrayBase<S, Ix2>> for ArrayBase<S, Ix2>
where
    A: LapackScalar,
    S: DataMut<Elem = A>,
    Se: DataOwned<Elem = A::Real>,
{
    fn eigh(mut self, uplo: UPLO) -> Result<(ArrayBase<Se, Ix1>, ArrayBase<S, Ix2>)> {
        let s = A::eigh(true, self.square_layout()?, uplo, self.as_allocated_mut()?)?;
        Ok((ArrayBase::from_vec(s), self))
    }
}

impl<'a, A, S, Se, So> Eigh<ArrayBase<Se, Ix1>, ArrayBase<So, Ix2>> for &'a ArrayBase<S, Ix2>
    where A: LapackScalar + Copy,
          S: Data<Elem = A>,
          Se: DataOwned<Elem = A::Real>,
          So: DataOwned<Elem = A> + DataMut
{
    fn eigh(self, uplo: UPLO) -> Result<(ArrayBase<Se, Ix1>, ArrayBase<So, Ix2>)> {
        let mut a = replicate(self);
        let s = A::eigh(true, a.square_layout()?, uplo, a.as_allocated_mut()?)?;
        Ok((ArrayBase::from_vec(s), a))
    }
}

impl<'a, A, S, Se> Eigh<ArrayBase<Se, Ix1>, &'a mut ArrayBase<S, Ix2>> for &'a mut ArrayBase<S, Ix2>
    where A: LapackScalar,
          S: DataMut<Elem = A>,
          Se: DataOwned<Elem = A::Real>
{
    fn eigh(mut self, uplo: UPLO) -> Result<(ArrayBase<Se, Ix1>, &'a mut ArrayBase<S, Ix2>)> {
        let s = A::eigh(true, self.square_layout()?, uplo, self.as_allocated_mut()?)?;
        Ok((ArrayBase::from_vec(s), self))
    }
}

pub trait EigValsh<EigVal> {
    fn eigvalsh(self, UPLO) -> Result<EigVal>;
}

impl<A, S, Se> EigValsh<ArrayBase<Se, Ix1>> for ArrayBase<S, Ix2>
where
    A: LapackScalar,
    S: DataMut<Elem = A>,
    Se: DataOwned<Elem = A::Real>,
{
    fn eigvalsh(mut self, uplo: UPLO) -> Result<ArrayBase<Se, Ix1>> {
        let s = A::eigh(false, self.square_layout()?, uplo, self.as_allocated_mut()?)?;
        Ok(ArrayBase::from_vec(s))
    }
}

impl<'a, A, S, Se> EigValsh<ArrayBase<Se, Ix1>> for &'a ArrayBase<S, Ix2>
where
    A: LapackScalar + Copy,
    S: Data<Elem = A>,
    Se: DataOwned<Elem = A::Real>,
{
    fn eigvalsh(self, uplo: UPLO) -> Result<ArrayBase<Se, Ix1>> {
        let mut a = self.to_owned();
        let s = A::eigh(false, a.square_layout()?, uplo, a.as_allocated_mut()?)?;
        Ok(ArrayBase::from_vec(s))
    }
}

impl<'a, A, S, Se> EigValsh<ArrayBase<Se, Ix1>> for &'a mut ArrayBase<S, Ix2>
where
    A: LapackScalar,
    S: DataMut<Elem = A>,
    Se: DataOwned<Elem = A::Real>,
{
    fn eigvalsh(mut self, uplo: UPLO) -> Result<ArrayBase<Se, Ix1>> {
        let s = A::eigh(true, self.square_layout()?, uplo, self.as_allocated_mut()?)?;
        Ok(ArrayBase::from_vec(s))
    }
}

pub trait SymmetricSqrt<Output> {
    fn ssqrt(self, UPLO) -> Result<Output>;
}

impl<A, S> SymmetricSqrt<ArrayBase<S, Ix2>> for ArrayBase<S, Ix2>
where
    A: Field,
    S: DataMut<Elem = A> + DataOwned,
{
    fn ssqrt(self, uplo: UPLO) -> Result<ArrayBase<S, Ix2>> {
        let (e, v): (Array1<A::Real>, _) = self.eigh(uplo)?;
        let e_sqrt = Array1::from_iter(e.iter().map(|r| AssociatedReal::inject(r.sqrt())));
        let ev: Array2<_> = e_sqrt.into_diagonal().op(&v.t());
        Ok(v.op(&ev))
    }
}
