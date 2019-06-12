//! Modified Gram-Schmit orthogonalizer

use super::*;
use crate::{generate::*, inner::*, norm::Norm};

/// Iterative orthogonalizer using modified Gram-Schmit procedure
#[derive(Debug, Clone)]
pub struct MGS<A> {
    /// Dimension of base space
    dimension: usize,
    /// Basis of spanned space
    q: Vec<Array1<A>>,
}

impl<A: Scalar + Lapack> MGS<A> {
    /// Create an empty orthogonalizer
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            q: Vec::new(),
        }
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

    fn decompose<S>(&self, a: &mut ArrayBase<S, Ix1>) -> Array1<A>
    where
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

    fn coeff<S>(&self, a: ArrayBase<S, Ix1>) -> Array1<A>
    where
        A: Lapack,
        S: Data<Elem = A>,
    {
        let mut a = a.into_owned();
        self.decompose(&mut a)
    }

    fn append<S>(&mut self, a: ArrayBase<S, Ix1>, rtol: A::Real) -> Result<Array1<A>, Array1<A>>
    where
        A: Lapack,
        S: Data<Elem = A>,
    {
        let mut a = a.into_owned();
        let coef = self.decompose(&mut a);
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

/// Online QR decomposition using modified Gram-Schmit algorithm
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
