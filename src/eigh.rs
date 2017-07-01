//! Eigenvalue decomposition for Hermite matrices

use ndarray::*;

use super::convert::*;
use super::diagonal::*;
use super::error::*;
use super::layout::*;
use super::operator::*;
use super::types::*;

use lapack_traits::LapackScalar;
pub use lapack_traits::UPLO;

/// Eigenvalue decomposition of Hermite matrix
pub trait Eigh {
    type EigVal;
    type EigVec;
    fn eigh(&self, UPLO) -> Result<(Self::EigVal, Self::EigVec)>;
}

/// Eigenvalue decomposition of Hermite matrix
pub trait EighMut {
    type EigVal;
    fn eigh_mut(&mut self, UPLO) -> Result<(Self::EigVal, &mut Self)>;
}

/// Eigenvalue decomposition of Hermite matrix
pub trait EighInto: Sized {
    type EigVal;
    fn eigh_into(self, UPLO) -> Result<(Self::EigVal, Self)>;
}

impl<A, S> EighInto for ArrayBase<S, Ix2>
where
    A: LapackScalar,
    S: DataMut<Elem = A>,
{
    type EigVal = Array1<A::Real>;

    fn eigh_into(mut self, uplo: UPLO) -> Result<(Self::EigVal, Self)> {
        let s = A::eigh(true, self.square_layout()?, uplo, self.as_allocated_mut()?)?;
        Ok((Array::from_vec(s), self))
    }
}

impl<A, S> Eigh for ArrayBase<S, Ix2>
where
    A: LapackScalar + Copy,
    S: Data<Elem = A>,
{
    type EigVal = Array1<A::Real>;
    type EigVec = Array2<A>;

    fn eigh(&self, uplo: UPLO) -> Result<(Self::EigVal, Self::EigVec)> {
        let mut a = replicate(self);
        let s = A::eigh(true, a.square_layout()?, uplo, a.as_allocated_mut()?)?;
        Ok((ArrayBase::from_vec(s), a))
    }
}

impl<A, S> EighMut for ArrayBase<S, Ix2>
where
    A: LapackScalar,
    S: DataMut<Elem = A>,
{
    type EigVal = Array1<A::Real>;

    fn eigh_mut(&mut self, uplo: UPLO) -> Result<(Self::EigVal, &mut Self)> {
        let s = A::eigh(true, self.square_layout()?, uplo, self.as_allocated_mut()?)?;
        Ok((ArrayBase::from_vec(s), self))
    }
}

/// Calculate eigenvalues without eigenvectors
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

/// Calculate symmetric square-root matrix using `eigh`
pub trait SymmetricSqrt {
    type Output;
    fn ssqrt(&self, UPLO) -> Result<Self::Output>;
}

impl<A, S> SymmetricSqrt for ArrayBase<S, Ix2>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    type Output = Array2<A>;

    fn ssqrt(&self, uplo: UPLO) -> Result<Self::Output> {
        let (e, v) = self.eigh(uplo)?;
        let e_sqrt = Array1::from_iter(e.iter().map(|r| AssociatedReal::inject(r.sqrt())));
        let ev = e_sqrt.into_diagonal().op(&v.t());
        Ok(v.op(&ev))
    }
}

/// Calculate symmetric square-root matrix using `eigh`
pub trait SymmetricSqrtInto {
    type Output;
    fn ssqrt_into(self, UPLO) -> Result<Self::Output>;
}

impl<A, S> SymmetricSqrtInto for ArrayBase<S, Ix2>
where
    A: Scalar,
    S: DataMut<Elem = A> + DataOwned,
{
    type Output = Array2<A>;

    fn ssqrt_into(self, uplo: UPLO) -> Result<Self::Output> {
        let (e, v) = self.eigh_into(uplo)?;
        let e_sqrt = Array1::from_iter(e.iter().map(|r| AssociatedReal::inject(r.sqrt())));
        let ev = e_sqrt.into_diagonal().op(&v.t());
        Ok(v.op(&ev))
    }
}
