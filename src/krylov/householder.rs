use super::*;
use crate::{inner::*, norm::*};
use num_traits::Zero;

/// Iterative orthogonalizer using Householder reflection
#[derive(Debug, Clone)]
pub struct Householder<A: Scalar> {
    /// Dimension of orthogonalizer
    dim: usize,

    /// Store Householder reflector.
    ///
    /// The coefficient is copied into another array, and this does not contain
    v: Vec<Array1<A>>,
}

impl<A: Scalar> Householder<A> {
    /// Create a new orthogonalizer
    pub fn new(dim: usize) -> Self {
        Householder { dim, v: Vec::new() }
    }

    /// Take a Reflection `P = I - 2ww^T`
    fn reflect<S: DataMut<Elem = A>>(&self, k: usize, a: &mut ArrayBase<S, Ix1>) {
        assert!(k < self.v.len());
        assert_eq!(a.len(), self.dim, "Input array size mismaches to the dimension");

        let w = self.v[k].slice(s![k..]);
        let mut a_slice = a.slice_mut(s![k..]);
        let c = A::from(2.0).unwrap() * w.inner(&a_slice);
        for l in 0..self.dim - k {
            a_slice[l] -= c * w[l];
        }
    }
}

impl<A: Scalar + Lapack> Orthogonalizer for Householder<A> {
    type Elem = A;

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
        if self.is_full() {
            return Zero::zero();
        }
        // residual norm
        a.slice(s![self.len()..]).norm_l2()
    }

    fn append<S>(&mut self, mut a: ArrayBase<S, Ix1>, rtol: A::Real) -> Result<Array1<A>, Array1<A>>
    where
        S: DataMut<Elem = A>,
    {
        assert_eq!(a.len(), self.dim);
        let k = self.len();
        let alpha = self.orthogonalize(&mut a);
        let mut coef = Array::zeros(k + 1);
        for i in 0..k {
            coef[i] = a[i];
        }
        if alpha < rtol {
            // linearly dependent
            coef[k] = A::from_real(alpha);
            return Err(coef);
        }

        // Add reflector
        assert!(k < a.len()); // this must hold because `alpha == 0` if k >= a.len()
        let alpha = if a[k].abs() > Zero::zero() {
            a[k].mul_real(alpha / a[k].abs())
        } else {
            A::from_real(alpha)
        };
        coef[k] = alpha;

        a[k] -= alpha;
        let norm = a.slice(s![k..]).norm_l2();
        azip!(mut a (a.slice_mut(s![..k])) in { *a = Zero::zero() }); // this can be omitted
        azip!(mut a (a.slice_mut(s![k..])) in { *a = a.div_real(norm) });
        self.v.push(a.into_owned());
        Ok(coef)
    }

    fn get_q(&self) -> Q<A> {
        assert!(self.len() > 0);
        let mut a = Array::zeros((self.dim(), self.len()));
        for (i, mut col) in a.axis_iter_mut(Axis(1)).enumerate() {
            col[i] = A::one();
            for l in (0..self.len()).rev() {
                self.reflect(l, &mut col);
            }
        }
        a
    }
}
