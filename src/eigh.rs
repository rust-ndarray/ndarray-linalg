//! Eigenvalue decomposition for Hermite matrices

use ndarray::*;

use crate::diagonal::*;
use crate::error::*;
use crate::layout::*;
use crate::operator::LinearOperator;
use crate::types::*;
use crate::UPLO;
use std::iter::FromIterator;

/// Eigenvalue decomposition of Hermite matrix reference
pub trait Eigh {
    type EigVal;
    type EigVec;
    fn eigh(&self, uplo: UPLO) -> Result<(Self::EigVal, Self::EigVec)>;
}

/// Eigenvalue decomposition of mutable reference of Hermite matrix
pub trait EighInplace {
    type EigVal;
    fn eigh_inplace(&mut self, uplo: UPLO) -> Result<(Self::EigVal, &mut Self)>;
}

/// Eigenvalue decomposition of Hermite matrix
pub trait EighInto: Sized {
    type EigVal;
    fn eigh_into(self, uplo: UPLO) -> Result<(Self::EigVal, Self)>;
}

impl<A, S> EighInto for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
{
    type EigVal = Array1<A::Real>;

    fn eigh_into(mut self, uplo: UPLO) -> Result<(Self::EigVal, Self)> {
        let (val, _) = self.eigh_inplace(uplo)?;
        Ok((val, self))
    }
}

impl<A, S> Eigh for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
{
    type EigVal = Array1<A::Real>;
    type EigVec = Array2<A>;

    fn eigh(&self, uplo: UPLO) -> Result<(Self::EigVal, Self::EigVec)> {
        let a = self.to_owned();
        a.eigh_into(uplo)
    }
}

impl<A, S> EighInplace for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
{
    type EigVal = Array1<A::Real>;

    fn eigh_inplace(&mut self, uplo: UPLO) -> Result<(Self::EigVal, &mut Self)> {
        let layout = self.square_layout()?;
        // XXX Force layout to be Fortran (see #146)
        match layout {
            MatrixLayout::C(_) => self.swap_axes(0, 1),
            MatrixLayout::F(_) => {}
        }
        let s = unsafe { A::eigh(true, self.square_layout()?, uplo, self.as_allocated_mut()?)? };
        Ok((ArrayBase::from(s), self))
    }
}

/// Calculate eigenvalues without eigenvectors
pub trait EigValsh {
    type EigVal;
    fn eigvalsh(&self, uplo: UPLO) -> Result<Self::EigVal>;
}

/// Calculate eigenvalues without eigenvectors
pub trait EigValshInto {
    type EigVal;
    fn eigvalsh_into(self, uplo: UPLO) -> Result<Self::EigVal>;
}

/// Calculate eigenvalues without eigenvectors
pub trait EigValshInplace {
    type EigVal;
    fn eigvalsh_inplace(&mut self, uplo: UPLO) -> Result<Self::EigVal>;
}

impl<A, S> EigValshInto for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
{
    type EigVal = Array1<A::Real>;

    fn eigvalsh_into(mut self, uplo: UPLO) -> Result<Self::EigVal> {
        self.eigvalsh_inplace(uplo)
    }
}

impl<A, S> EigValsh for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
{
    type EigVal = Array1<A::Real>;

    fn eigvalsh(&self, uplo: UPLO) -> Result<Self::EigVal> {
        let a = self.to_owned();
        a.eigvalsh_into(uplo)
    }
}

impl<A, S> EigValshInplace for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
{
    type EigVal = Array1<A::Real>;

    fn eigvalsh_inplace(&mut self, uplo: UPLO) -> Result<Self::EigVal> {
        let s = unsafe { A::eigh(true, self.square_layout()?, uplo, self.as_allocated_mut()?)? };
        Ok(ArrayBase::from(s))
    }
}

/// Calculate symmetric square-root matrix using `eigh`
pub trait SymmetricSqrt {
    type Output;
    fn ssqrt(&self, uplo: UPLO) -> Result<Self::Output>;
}

impl<A, S> SymmetricSqrt for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
{
    type Output = Array2<A>;

    fn ssqrt(&self, uplo: UPLO) -> Result<Self::Output> {
        let a = self.to_owned();
        a.ssqrt_into(uplo)
    }
}

/// Calculate symmetric square-root matrix using `eigh`
pub trait SymmetricSqrtInto {
    type Output;
    fn ssqrt_into(self, uplo: UPLO) -> Result<Self::Output>;
}

impl<A, S> SymmetricSqrtInto for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A> + DataOwned,
{
    type Output = Array2<A>;

    fn ssqrt_into(self, uplo: UPLO) -> Result<Self::Output> {
        let (e, v) = self.eigh_into(uplo)?;
        let e_sqrt = Array::from_iter(e.iter().map(|r| Scalar::from_real(r.sqrt())));
        let ev = e_sqrt.into_diagonal().apply2(&v.t());
        Ok(v.apply2(&ev))
    }
}
