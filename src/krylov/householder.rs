use super::*;
use crate::{inner::*, norm::*};

/// Iterative orthogonalizer using Householder reflection
#[derive(Debug, Clone)]
pub struct Householder<A> {
    dim: usize,
    v: Vec<Array1<A>>,
}

impl<A: Scalar> Householder<A> {
    pub fn new(dim: usize) -> Self {
        Householder { dim, v: Vec::new() }
    }

    /// Take a Reflection `P = I - 2ww^T`
    fn reflect<S: DataMut<Elem = A>>(&self, k: usize, a: &mut ArrayBase<S, Ix1>) {
        assert!(k < self.v.len());
        assert_eq!(a.len(), self.dim);
        let w = self.v[k].slice(s![k..]);
        let c = A::from(2.0).unwrap() * w.inner(&a.slice(s![k..]));
        for l in k..self.dim {
            a[l] -= c * w[l];
        }
    }
}

impl<A: Scalar + Lapack> Orthogonalizer for Householder<A> {
    type Elem = A;

    fn new(dim: usize) -> Self {
        Self { dim, v: Vec::new() }
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn len(&self) -> usize {
        self.v.len()
    }

    fn orthogonalize<S>(&self, a: &mut ArrayBase<S, Ix1>) -> A::Real
    where
        S: DataMut<Elem = A>,
    {
        for k in 0..self.len() {
            self.reflect(k, a);
        }
        // residual norm
        a.slice(s![self.len()..]).norm_l2()
    }

    fn append<S>(&mut self, mut a: ArrayBase<S, Ix1>, rtol: A::Real) -> Result<Array1<A>, Array1<A>>
    where
        S: DataMut<Elem = A>,
    {
        let residual = self.orthogonalize(&mut a);
        let coef = a.slice(s![..self.len()]).into_owned();
        if residual < rtol {
            return Err(coef);
        }
        self.v.push(a.into_owned());
        Ok(coef)
    }

    fn get_q(&self) -> Q<A> {
        unimplemented!()
    }
}
