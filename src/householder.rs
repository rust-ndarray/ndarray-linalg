use crate::{inner::*, norm::*, types::*};
use ndarray::*;

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

    pub fn len(&self) -> usize {
        self.v.len()
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

    /// Orghotonalize input vector by reflectors
    ///
    /// Panic
    /// -------
    /// - if the size of the input array mismaches to the dimension
    pub fn orthogonalize<S>(&self, a: &mut ArrayBase<S, Ix1>) -> A::Real
    where
        A: Lapack,
        S: DataMut<Elem = A>,
    {
        for k in 0..self.len() {
            self.reflect(k, a);
        }
        // residual norm
        a.slice(s![self.len()..]).norm_l2()
    }

    /// Orghotonalize input vector by reflectors
    ///
    /// ```rust
    /// # use ndarray::*;
    /// # use ndarray_linalg::*;
    /// let mut mgs = arnoldi::MGS::new(3);
    /// let coef = mgs.append(array![0.0, 1.0, 0.0], 1e-9).unwrap();
    /// close_l2(&coef, &array![1.0], 1e-9).unwrap();
    ///
    /// let coef = mgs.append(array![1.0, 1.0, 0.0], 1e-9).unwrap();
    /// close_l2(&coef, &array![1.0, 1.0], 1e-9).unwrap();
    ///
    /// assert!(mgs.append(array![1.0, 2.0, 0.0], 1e-9).is_err());  // Fail if the vector is linearly dependend
    ///
    /// if let Err(coef) = mgs.append(array![1.0, 2.0, 0.0], 1e-9) {
    ///     close_l2(&coef, &array![2.0, 1.0, 0.0], 1e-9).unwrap(); // You can get coefficients of dependent vector
    /// }
    /// ```
    ///
    /// Panic
    /// -------
    /// - if the size of the input array mismaches to the dimension
    ///
    pub fn append<S>(&mut self, mut a: ArrayBase<S, Ix1>, rtol: A::Real) -> Result<Array1<A>, Array1<A>>
    where
        A: Lapack,
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
}
