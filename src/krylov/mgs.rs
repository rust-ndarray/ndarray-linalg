//! Modified Gram-Schmit orthogonalizer

use super::*;
use crate::{generate::*, inner::*, norm::Norm};

/// Iterative orthogonalizer using modified Gram-Schmit procedure
///
/// ```rust
/// # use ndarray::*;
/// # use ndarray_linalg::{krylov::*, *};
/// let mut mgs = MGS::new(3);
/// let coef = mgs.append(array![0.0, 1.0, 0.0], 1e-9).unwrap();
/// close_l2(&coef, &array![1.0], 1e-9);
///
/// let coef = mgs.append(array![1.0, 1.0, 0.0], 1e-9).unwrap();
/// close_l2(&coef, &array![1.0, 1.0], 1e-9);
///
/// // Fail if the vector is linearly dependent
/// assert!(mgs.append(array![1.0, 2.0, 0.0], 1e-9).is_err());
///
/// // You can get coefficients of dependent vector
/// if let Err(coef) = mgs.append(array![1.0, 2.0, 0.0], 1e-9) {
///     close_l2(&coef, &array![2.0, 1.0, 0.0], 1e-9);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct MGS<A> {
    /// Dimension of base space
    dimension: usize,
    /// Basis of spanned space
    q: Vec<Array1<A>>,
}

impl<A: Scalar> MGS<A> {
    /// Create an empty orthogonalizer
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            q: Vec::new(),
        }
    }

    /// Orthogonalize given vector against to the current basis
    ///
    /// - Returned array is coefficients and residual norm
    /// - `a` will contain the residual vector
    ///
    fn orthogonalize<S>(&self, a: &mut ArrayBase<S, Ix1>) -> Array1<A>
    where
        A: Lapack,
        S: DataMut<Elem = A>,
    {
        assert_eq!(a.len(), self.dim());
        let mut coef = Array1::zeros(self.len() + 1);
        for i in 0..self.len() {
            let q = &self.q[i];
            let c = q.inner(&a);
            azip!(mut a (&mut *a), q (q) in { *a = *a - c * q } );
            coef[i] = c;
        }
        let nrm = a.norm_l2();
        coef[self.len()] = A::from_real(nrm);
        coef
    }
}

impl<A: Scalar + Lapack> Orthogonalizer for MGS<A> {
    type Elem = A;

    fn dim(&self) -> usize {
        self.dimension
    }

    fn len(&self) -> usize {
        self.q.len()
    }

    fn coeff<S>(&self, a: ArrayBase<S, Ix1>) -> Array1<A>
    where
        A: Lapack,
        S: Data<Elem = A>,
    {
        let mut a = a.into_owned();
        self.orthogonalize(&mut a)
    }

    fn append<S>(&mut self, a: ArrayBase<S, Ix1>, rtol: A::Real) -> Result<Array1<A>, Array1<A>>
    where
        A: Lapack,
        S: Data<Elem = A>,
    {
        let mut a = a.into_owned();
        let coef = self.orthogonalize(&mut a);
        let nrm = coef[coef.len() - 1].re();
        if nrm < rtol {
            // Linearly dependent
            return Err(coef);
        }
        azip!(mut a in { *a = *a / A::from_real(nrm) });
        self.q.push(a);
        Ok(coef)
    }

    fn get_q(&self) -> Q<A> {
        hstack(&self.q).unwrap()
    }
}

/// Online QR decomposition of vectors using modified Gram-Schmit algorithm
pub fn mgs<A, S>(
    iter: impl Iterator<Item = ArrayBase<S, Ix1>>,
    dim: usize,
    rtol: A::Real,
    strategy: Strategy,
) -> (Q<A>, R<A>)
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
{
    let mgs = MGS::new(dim);
    qr(iter, mgs, rtol, strategy)
}
