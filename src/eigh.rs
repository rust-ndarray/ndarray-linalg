
use ndarray::*;

use super::error::*;
use super::layout::*;

use impl2::{LapackScalar, UPLO};

pub trait Eigh<EigVal, EigVec> {
    fn eigh(self, UPLO) -> Result<(EigVal, EigVec)>;
}

impl<A, S, Se> Eigh<ArrayBase<Se, Ix1>, ArrayBase<S, Ix2>> for ArrayBase<S, Ix2>
    where A: LapackScalar,
          S: DataMut<Elem = A>,
          Se: DataOwned<Elem = A::Real>
{
    fn eigh(mut self, uplo: UPLO) -> Result<(ArrayBase<Se, Ix1>, ArrayBase<S, Ix2>)> {
        let s = A::eigh(true, self.square_layout()?, uplo, self.as_allocated_mut()?)?;
        Ok((ArrayBase::from_vec(s), self))
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
