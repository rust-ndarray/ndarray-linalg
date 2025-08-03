//! Eigenvalue decomposition for non-symmetric square matrices

use crate::error::*;
use crate::layout::*;
use crate::types::*;
pub use lax::GeneralizedEigenvalue;
use ndarray::*;

#[cfg_attr(doc, katexit::katexit)]
/// Eigenvalue decomposition of general matrix reference
pub trait Eig {
    /// EigVec is the right eivenvector
    type EigVal;
    type EigVec;
    /// Calculate eigenvalues with the right eigenvector
    ///
    /// $$ A u_i = \lambda_i u_i $$
    ///
    /// ```
    /// use ndarray::*;
    /// use ndarray_linalg::*;
    ///
    /// let a: Array2<f64> = array![
    ///     [-1.01,  0.86, -4.60,  3.31, -4.81],
    ///     [ 3.98,  0.53, -7.04,  5.29,  3.55],
    ///     [ 3.30,  8.26, -3.89,  8.20, -1.51],
    ///     [ 4.43,  4.96, -7.66, -7.33,  6.18],
    ///     [ 7.31, -6.43, -6.16,  2.47,  5.58],
    /// ];
    /// let (eigs, vecs) = a.eig().unwrap();
    ///
    /// let a = a.map(|v| v.as_c());
    /// for (&e, vec) in eigs.iter().zip(vecs.axis_iter(Axis(1))) {
    ///     let ev = vec.map(|v| v * e);
    ///     let av = a.dot(&vec);
    ///     assert_close_l2!(&av, &ev, 1e-5);
    /// }
    /// ```
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
        let (s, t) = A::eig(true, layout, a.as_allocated_mut()?)?;
        let n = layout.len() as usize;
        Ok((
            ArrayBase::from(s),
            Array2::from_shape_vec((n, n).f(), t).unwrap(),
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
    S: Data<Elem = A>,
{
    type EigVal = Array1<A::Complex>;

    fn eigvals(&self) -> Result<Self::EigVal> {
        let mut a = self.to_owned();
        let (s, _) = A::eig(false, a.square_layout()?, a.as_allocated_mut()?)?;
        Ok(ArrayBase::from(s))
    }
}

#[cfg_attr(doc, katexit::katexit)]
/// Eigenvalue decomposition of general matrix reference
pub trait EigGeneralized {
    /// EigVec is the right eivenvector
    type EigVal;
    type EigVec;
    type Real;
    /// Calculate eigenvalues with the right eigenvector
    ///
    /// $$ A u_i = \lambda_i B u_i $$
    ///
    /// ```
    /// use ndarray::*;
    /// use ndarray_linalg::*;
    ///
    /// let a: Array2<f64> = array![
    ///     [-1.01,  0.86, -4.60,  3.31, -4.81],
    ///     [ 3.98,  0.53, -7.04,  5.29,  3.55],
    ///     [ 3.30,  8.26, -3.89,  8.20, -1.51],
    ///     [ 4.43,  4.96, -7.66, -7.33,  6.18],
    ///     [ 7.31, -6.43, -6.16,  2.47,  5.58],
    /// ];
    /// let b: Array2<f64> = array![
    ///     [ 1.23, -4.56,  7.89,  0.12, -3.45],
    ///     [ 6.78, -9.01,  2.34, -5.67,  8.90],
    ///     [-1.11,  3.33, -6.66,  9.99, -2.22],
    ///     [ 4.44, -7.77,  0.00,  1.11,  5.55],
    ///     [-8.88,  6.66, -3.33,  2.22, -9.99],
    /// ];
    /// let (geneigs, vecs) = (a.clone(), b.clone()).eig_generalized(None).unwrap();
    ///
    /// let a = a.map(|v| v.as_c());
    /// let b = b.map(|v| v.as_c());
    /// for (ge, vec) in geneigs.iter().zip(vecs.axis_iter(Axis(1))) {
    ///     if let GeneralizedEigenvalue::Finite(e, _) = ge {
    ///         let ebv = b.dot(&vec).map(|v| v * e);
    ///         let av = a.dot(&vec);
    ///         assert_close_l2!(&av, &ebv, 1e-5);
    ///     }
    /// }
    /// ```
    ///
    /// # Arguments
    ///
    /// * `thresh_opt` - An optional threshold for determining approximate zero |β| values when
    /// computing the eigenvalues as α/β. If `None`, no approximate comparisons to zero will be
    /// made.
    fn eig_generalized(
        &self,
        thresh_opt: Option<Self::Real>,
    ) -> Result<(Self::EigVal, Self::EigVec)>;
}

impl<A, S> EigGeneralized for (ArrayBase<S, Ix2>, ArrayBase<S, Ix2>)
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
{
    type EigVal = Array1<GeneralizedEigenvalue<A::Complex>>;
    type EigVec = Array2<A::Complex>;
    type Real = A::Real;

    fn eig_generalized(
        &self,
        thresh_opt: Option<Self::Real>,
    ) -> Result<(Self::EigVal, Self::EigVec)> {
        let (mut a, mut b) = (self.0.to_owned(), self.1.to_owned());
        let layout = a.square_layout()?;
        let (s, t) = A::eig_generalized(
            true,
            layout,
            a.as_allocated_mut()?,
            b.as_allocated_mut()?,
            thresh_opt,
        )?;
        let n = layout.len() as usize;
        Ok((
            ArrayBase::from(s),
            Array2::from_shape_vec((n, n).f(), t).unwrap(),
        ))
    }
}
