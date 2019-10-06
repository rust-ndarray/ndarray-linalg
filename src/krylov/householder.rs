//! Householder reflection
//!
//! - [Householder transformation - Wikipedia](https://en.wikipedia.org/wiki/Householder_transformation)
//!

use super::*;
use crate::{inner::*, norm::*};
use num_traits::One;

/// Calc a reflactor `w` from a vector `x`
pub fn calc_reflector<A, S>(x: &mut ArrayBase<S, Ix1>)
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
{
    assert!(x.len() > 0);
    let norm = x.norm_l2();
    let alpha = -x[0].mul_real(norm / x[0].abs());
    x[0] -= alpha;
    let inv_rev_norm = A::Real::one() / x.norm_l2();
    // azip!(mut a(x) in { *a = a.mul_real(inv_rev_norm)});
    azip!((a in x) *a = a.mul_real(inv_rev_norm));
}

/// Take a reflection `P = I - 2ww^T`
///
/// Panic
/// ------
/// - if the size of `w` and `a` mismaches
pub fn reflect<A, S1, S2>(w: &ArrayBase<S1, Ix1>, a: &mut ArrayBase<S2, Ix1>)
where
    A: Scalar + Lapack,
    S1: Data<Elem = A>,
    S2: DataMut<Elem = A>,
{
    assert_eq!(w.len(), a.len());
    let n = a.len();
    let c = A::from(2.0).unwrap() * w.inner(&a);
    for l in 0..n {
        a[l] -= c * w[l];
    }
}

/// Iterative orthogonalizer using Householder reflection
#[derive(Debug, Clone)]
pub struct Householder<A: Scalar> {
    /// Dimension of orthogonalizer
    dim: usize,

    /// Store Householder reflector.
    ///
    /// The coefficient is copied into another array, and this does not contain
    v: Vec<Array1<A>>,

    /// Tolerance
    tol: A::Real,
}

impl<A: Scalar + Lapack> Householder<A> {
    /// Create a new orthogonalizer
    pub fn new(dim: usize, tol: A::Real) -> Self {
        Householder {
            dim,
            v: Vec::new(),
            tol,
        }
    }

    /// Take a Reflection `P = I - 2ww^T`
    fn fundamental_reflection<S>(&self, k: usize, a: &mut ArrayBase<S, Ix1>)
    where
        S: DataMut<Elem = A>,
    {
        assert!(k < self.v.len());
        assert_eq!(a.len(), self.dim, "Input array size mismaches to the dimension");
        reflect(&self.v[k].slice(s![k..]), &mut a.slice_mut(s![k..]));
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

    /// Compose coefficients array using reflected vector
    fn compose_coefficients<S>(&self, a: &ArrayBase<S, Ix1>) -> Coefficients<A>
    where
        S: Data<Elem = A>,
    {
        let k = self.len();
        let res = a.slice(s![k..]).norm_l2();
        let mut c = Array1::zeros(k + 1);
        azip!((c in c.slice_mut(s![..k]), &a in a.slice(s![..k])) *c = a);
        if k < a.len() {
            let ak = a[k];
            c[k] = -ak.mul_real(res / ak.abs());
        } else {
            c[k] = A::from_real(res);
        }
        c
    }

    /// Construct the residual vector from reflected vector
    fn construct_residual<S>(&self, a: &mut ArrayBase<S, Ix1>)
    where
        S: DataMut<Elem = A>,
    {
        let k = self.len();
        azip!((a in a.slice_mut(s![..k])) *a = A::zero());
        self.backward_reflection(a);
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

    fn tolerance(&self) -> A::Real {
        self.tol
    }

    fn decompose<S>(&self, a: &mut ArrayBase<S, Ix1>) -> Array1<A>
    where
        S: DataMut<Elem = A>,
    {
        self.forward_reflection(a);
        let coef = self.compose_coefficients(a);
        self.construct_residual(a);
        coef
    }

    fn coeff<S>(&self, a: ArrayBase<S, Ix1>) -> Array1<A>
    where
        S: Data<Elem = A>,
    {
        let mut a = a.into_owned();
        self.forward_reflection(&mut a);
        self.compose_coefficients(&a)
    }

    fn div_append<S>(&mut self, a: &mut ArrayBase<S, Ix1>) -> AppendResult<A>
    where
        S: DataMut<Elem = A>,
    {
        assert_eq!(a.len(), self.dim);
        let k = self.len();
        self.forward_reflection(a);
        let coef = self.compose_coefficients(a);
        if coef[k].abs() < self.tol {
            return AppendResult::Dependent(coef);
        }
        calc_reflector(&mut a.slice_mut(s![k..]));
        self.v.push(a.to_owned());
        self.construct_residual(a);
        AppendResult::Added(coef)
    }

    fn append<S>(&mut self, a: ArrayBase<S, Ix1>) -> AppendResult<A>
    where
        S: Data<Elem = A>,
    {
        assert_eq!(a.len(), self.dim);
        let mut a = a.into_owned();
        let k = self.len();
        self.forward_reflection(&mut a);
        let coef = self.compose_coefficients(&a);
        if coef[k].abs() < self.tol {
            return AppendResult::Dependent(coef);
        }
        calc_reflector(&mut a.slice_mut(s![k..]));
        self.v.push(a.to_owned());
        AppendResult::Added(coef)
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

/// Online QR decomposition using Householder reflection
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
    let h = Householder::new(dim, rtol);
    qr(iter, h, strategy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert::*;
    use num_traits::Zero;

    #[test]
    fn check_reflector() {
        let mut a = array![c64::new(1.0, 1.0), c64::new(1.0, 0.0), c64::new(0.0, 1.0)];
        let mut w = a.clone();
        calc_reflector(&mut w);
        reflect(&w, &mut a);
        close_l2(
            &a,
            &array![-c64::new(2.0.sqrt(), 2.0.sqrt()), c64::zero(), c64::zero()],
            1e-9,
        );
    }
}
