//! Eigendecomposition for Hermitian matrices.
//!
//! For a Hermitian matrix `A`, this solves the eigenvalue problem `A V = V D`
//! for `D` and `V`, where `D` is the diagonal matrix of eigenvalues in
//! ascending order and `V` is the orthonormal matrix of corresponding
//! eigenvectors.
//!
//! For a pair of Hermitian matrices `A` and `B` where `B` is also positive
//! definite, this solves the generalized eigenvalue problem `A V = B V D`,
//! where `D` is the diagonal matrix of generalized eigenvalues in ascending
//! order and `V` is the matrix of corresponding generalized eigenvectors. The
//! matrix `V` is normalized such that `V^H B V = I`.
//!
//! # Example
//!
//! Find the eigendecomposition of a Hermitian (or real symmetric) matrix.
//!
//! ```
//! use approx::assert_abs_diff_eq;
//! use ndarray::{array, Array2};
//! use ndarray_linalg::{Eigh, UPLO};
//!
//! let a: Array2<f64> = array![
//!     [2., 1.],
//!     [1., 2.],
//! ];
//! let (eigvals, eigvecs) = a.eigh(UPLO::Lower)?;
//! assert_abs_diff_eq!(eigvals, array![1., 3.]);
//! assert_abs_diff_eq!(
//!     a.dot(&eigvecs),
//!     eigvecs.dot(&Array2::from_diag(&eigvals)),
//! );
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use ndarray::*;

use crate::diagonal::*;
use crate::error::*;
use crate::layout::*;
use crate::operator::LinearOperator;
use crate::types::*;
use crate::UPLO;

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

impl<A, S, S2> EighInto for (ArrayBase<S, Ix2>, ArrayBase<S2, Ix2>)
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
    S2: DataMut<Elem = A>,
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

impl<A, S, S2> Eigh for (ArrayBase<S, Ix2>, ArrayBase<S2, Ix2>)
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
    S2: Data<Elem = A>,
{
    type EigVal = Array1<A::Real>;
    type EigVec = (Array2<A>, Array2<A>);

    fn eigh(&self, uplo: UPLO) -> Result<(Self::EigVal, Self::EigVec)> {
        let (a, b) = (self.0.to_owned(), self.1.to_owned());
        (a, b).eigh_into(uplo)
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
            MatrixLayout::C { .. } => self.swap_axes(0, 1),
            MatrixLayout::F { .. } => {}
        }
        let s = A::eigh(true, self.square_layout()?, uplo, self.as_allocated_mut()?)?;
        Ok((ArrayBase::from(s), self))
    }
}

impl<A, S, S2> EighInplace for (ArrayBase<S, Ix2>, ArrayBase<S2, Ix2>)
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
    S2: DataMut<Elem = A>,
{
    type EigVal = Array1<A::Real>;

    /// Solves the generalized eigenvalue problem.
    ///
    /// # Panics
    ///
    /// Panics if the shapes of the matrices are different.
    fn eigh_inplace(&mut self, uplo: UPLO) -> Result<(Self::EigVal, &mut Self)> {
        assert_eq!(
            self.0.shape(),
            self.1.shape(),
            "The shapes of the matrices must be identical.",
        );
        let layout = self.0.square_layout()?;
        // XXX Force layout to be Fortran (see #146)
        match layout {
            MatrixLayout::C { .. } => self.0.swap_axes(0, 1),
            MatrixLayout::F { .. } => {}
        }

        let layout = self.1.square_layout()?;
        match layout {
            MatrixLayout::C { .. } => self.1.swap_axes(0, 1),
            MatrixLayout::F { .. } => {}
        }

        let s = A::eigh_generalized(
            true,
            self.0.square_layout()?,
            uplo,
            self.0.as_allocated_mut()?,
            self.1.as_allocated_mut()?,
        )?;

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
        let s = A::eigh(true, self.square_layout()?, uplo, self.as_allocated_mut()?)?;
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
