use crate::{generate::*, inner::*, norm::Norm, types::*};
use ndarray::*;

/// Iterative orthogonalizer using modified Gram-Schmit procedure
#[derive(Debug, Clone)]
pub struct MGS<A> {
    /// Dimension of base space
    dimension: usize,
    /// Basis of spanned space
    q: Vec<Array1<A>>,
}

/// Residual vector of orthogonalization
pub type Residual<S> = ArrayBase<S, Ix1>;
/// Residual vector of orthogonalization
pub type Coefficient<A> = Array1<A>;
/// Q-matrix (unitary matrix)
pub type Q<A> = Array2<A>;
/// R-matrix (upper triangle)
pub type R<A> = Array2<A>;

impl<A: Scalar> MGS<A> {
    /// Create empty linear space
    ///
    /// ```rust
    /// # use ndarray_linalg::*;
    /// const N: usize = 5;
    /// let mgs = arnoldi::MGS::<f32>::new(N);
    /// assert_eq!(mgs.dim(), N);
    /// assert_eq!(mgs.len(), 0);
    /// ```
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            q: Vec::new(),
        }
    }

    pub fn dim(&self) -> usize {
        self.dimension
    }

    pub fn len(&self) -> usize {
        self.q.len()
    }

    /// Orthogonalize given vector using current basis
    ///
    /// Panic
    /// -------
    /// - if the size of the input array mismaches to the dimension
    pub fn orthogonalize<S>(&self, mut a: ArrayBase<S, Ix1>) -> (Residual<S>, Coefficient<A>)
    where
        A: Lapack,
        S: DataMut<Elem = A>,
    {
        assert_eq!(a.len(), self.dim());
        let mut coef = Array1::zeros(self.len() + 1);
        for i in 0..self.len() {
            let q = &self.q[i];
            let c = q.inner(&a);
            azip!(mut a, q (q) in { *a = *a - c * q } );
            coef[i] = c;
        }
        let nrm = a.norm_l2();
        coef[self.len()] = A::from_real(nrm);
        (a, coef)
    }

    /// Add new vector if the residual is larger than relative tolerance
    ///
    /// Panic
    /// -------
    /// - if the size of the input array mismaches to the dimension
    ///
    /// ```rust
    /// # use ndarray::*;
    /// # use ndarray_linalg::*;
    /// let mut mgs = arnoldi::MGS::new(3);
    /// let coef = mgs.append(array![1.0, 0.0, 0.0], 1e-9).unwrap();
    /// close_l2(&coef, &array![1.0], 1e-9).unwrap();
    ///
    /// let coef = mgs.append(array![1.0, 1.0, 0.0], 1e-9).unwrap();
    /// close_l2(&coef, &array![1.0, 1.0], 1e-9).unwrap();
    ///
    /// assert!(mgs.append(array![1.0, 1.0, 0.0], 1e-9).is_none());  // Cannot append dependent vector
    /// ```
    pub fn append<S>(&mut self, a: ArrayBase<S, Ix1>, rtol: A::Real) -> Option<Coefficient<A>>
    where
        A: Lapack,
        S: Data<Elem = A>,
    {
        let a = a.into_owned();
        let (mut a, coef) = self.orthogonalize(a);
        let nrm = coef[coef.len() - 1].re();
        if nrm < rtol {
            // Linearly dependent
            return None;
        }
        azip!(mut a in { *a = *a / A::from_real(nrm) });
        self.q.push(a);
        Some(coef)
    }

    /// Get orthogonal basis as Q matrix
    pub fn get_q(&self) -> Q<A> {
        hstack(&self.q).unwrap()
    }
}

pub fn mgs<A, S>(iter: impl Iterator<Item = ArrayBase<S, Ix1>>, dim: usize, rtol: A::Real) -> (Q<A>, R<A>)
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
{
    let mut ortho = MGS::new(dim);
    let mut coefs = Vec::new();
    for a in iter {
        match ortho.append(a, rtol) {
            Some(coef) => coefs.push(coef),
            None => break,
        }
    }
    let m = coefs.len();
    let mut r = Array2::zeros((m, m));
    for i in 0..m {
        for j in 0..=i {
            r[(j, i)] = coefs[i][j];
        }
    }
    (ortho.get_q(), r)
}
