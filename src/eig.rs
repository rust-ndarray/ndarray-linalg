//! Eigenvalue decomposition for non-symmetric square matrices

use crate::error::*;
use crate::layout::*;
use crate::types::*;
use ndarray::*;

/// Eigenvalue decomposition of general matrix reference
pub trait Eig {
    /// EigVec is the right eivenvector
    type EigVal;
    type EigVec;
    /// Calculate eigenvalues with the right eigenvector
    fn eig(&self) -> Result<(Self::EigVal, Self::EigVec)>;
}

impl<A, S> Eig for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
{
    type EigVal = Array1<A::Complex>;
    type EigVec = Array2<A::Complex>;

    fn eig(&self) -> Result<(Self::EigVal, Self::EigVec)> {
        let mut a = self.to_owned();
        let layout = a.square_layout()?;
        let (s, t) = unsafe { A::eig(true, layout, a.as_allocated_mut()?)? };
        let (n, _) = layout.size();
        Ok((
            ArrayBase::from(s),
            ArrayBase::from(t).into_shape((n as usize, n as usize)).unwrap(),
        ))
    }
}

/// Calculate eigenvalues without eigenvectors
pub trait EigVals {
    type EigVal;
    fn eigvals(&self) -> Result<Self::EigVal>;
}

impl<A, S> EigVals for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
{
    type EigVal = Array1<A::Complex>;

    fn eigvals(&self) -> Result<Self::EigVal> {
        let mut a = self.to_owned();
        let (s, _) = unsafe { A::eig(true, a.square_layout()?, a.as_allocated_mut()?)? };
        Ok(ArrayBase::from(s))
    }
}
