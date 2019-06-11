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

impl<A: Scalar + Lapack> Householder<A> {
    /// Create a new orthogonalizer
    pub fn new(dim: usize) -> Self {
        Householder { dim, v: Vec::new() }
    }

    /// Take a Reflection `P = I - 2ww^T`
    fn fundamental_reflection<S>(&self, k: usize, a: &mut ArrayBase<S, Ix1>)
    where
        S: DataMut<Elem = A>,
    {
        assert!(k < self.v.len());
        assert_eq!(a.len(), self.dim, "Input array size mismaches to the dimension");

        let w = self.v[k].slice(s![k..]);
        let mut a_slice = a.slice_mut(s![k..]);
        let c = A::from(2.0).unwrap() * w.inner(&a_slice);
        for l in 0..self.dim - k {
            a_slice[l] -= c * w[l];
        }
    }

    /// Take forward reflection `P = P_l ... P_1`
    pub fn forward_reflection<S>(&self, a: &mut ArrayBase<S, Ix1>)
    where
        S: DataMut<Elem = A>,
    {
        assert!(a.len() == self.dim);
        let l = self.v.len();
        for k in 0..l {
            self.fundamental_reflection(k, a);
        }
    }

    /// Take backward reflection `P = P_1 ... P_l`
    pub fn backward_reflection<S>(&self, a: &mut ArrayBase<S, Ix1>)
    where
        S: DataMut<Elem = A>,
    {
        assert!(a.len() == self.dim);
        let l = self.v.len();
        for k in (0..l).rev() {
            self.fundamental_reflection(k, a);
        }
    }

    fn eval_residual<S>(&self, a: &ArrayBase<S, Ix1>) -> A::Real
    where
        S: Data<Elem = A>,
    {
        let l = self.v.len();
        if l == self.dim {
            Zero::zero()
        } else {
            a.slice(s![l..]).norm_l2()
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

    fn coeff<S>(&self, a: ArrayBase<S, Ix1>) -> Array1<A>
    where
        S: Data<Elem = A>,
    {
        let mut a = a.into_owned();
        self.forward_reflection(&mut a);
        let res = self.eval_residual(&a);
        let k = self.len();
        let mut c = Array1::zeros(k + 1);
        azip!(mut c(c.slice_mut(s![..k])), a(a.slice(s![..k])) in { *c = a });
        c[k] = A::from_real(res);
        c
    }

    fn append<S>(&mut self, mut a: ArrayBase<S, Ix1>, rtol: A::Real) -> Result<Array1<A>, Array1<A>>
    where
        S: DataMut<Elem = A>,
    {
        assert_eq!(a.len(), self.dim);
        let k = self.len();
        self.forward_reflection(&mut a);
        let alpha = self.eval_residual(&a);
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
            self.backward_reflection(&mut col);
        }
        a
    }
}

/// Online QR decomposition of vectors
pub fn householder<A, S>(
    iter: impl Iterator<Item = ArrayBase<S, Ix1>>,
    dim: usize,
    rtol: A::Real,
    strategy: Strategy,
) -> (Q<A>, R<A>)
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
{
    let h = Householder::new(dim);
    qr(iter, h, rtol, strategy)
}
