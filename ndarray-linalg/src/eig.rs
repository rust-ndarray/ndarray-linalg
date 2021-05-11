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
    S: DataMut<Elem = A>,
{
    type EigVal = Array1<A::Complex>;

    fn eigvals(&self) -> Result<Self::EigVal> {
        let mut a = self.to_owned();
        let (s, _) = A::eig(false, a.square_layout()?, a.as_allocated_mut()?)?;
        Ok(ArrayBase::from(s))
    }
}
