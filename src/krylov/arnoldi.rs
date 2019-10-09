//! Arnoldi iteration

use super::*;
use crate::{norm::Norm, operator::LinearOperator};
use num_traits::One;
use std::iter::*;

/// Execute Arnoldi iteration as Rust iterator
///
/// - [Arnoldi iteration - Wikipedia](https://en.wikipedia.org/wiki/Arnoldi_iteration)
///
pub struct Arnoldi<A, S, F, Ortho>
where
    A: Scalar,
    S: DataMut<Elem = A>,
    F: LinearOperator<Elem = A>,
    Ortho: Orthogonalizer<Elem = A>,
{
    a: F,
    /// Next vector (normalized `|v|=1`)
    v: ArrayBase<S, Ix1>,
    /// Orthogonalizer
    ortho: Ortho,
    /// Coefficients to be composed into H-matrix
    h: Vec<Array1<A>>,
}

impl<A, S, F, Ortho> Arnoldi<A, S, F, Ortho>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
    F: LinearOperator<Elem = A>,
    Ortho: Orthogonalizer<Elem = A>,
{
    /// Create an Arnoldi iterator from any linear operator `a`
    pub fn new(a: F, mut v: ArrayBase<S, Ix1>, mut ortho: Ortho) -> Self {
        assert_eq!(ortho.len(), 0);
        assert!(ortho.tolerance() < One::one());
        // normalize before append because |v| may be smaller than ortho.tolerance()
        let norm = v.norm_l2();
        azip!((v in &mut v)  *v = v.div_real(norm));
        ortho.append(v.view());
        Arnoldi {
            a,
            v,
            ortho,
            h: Vec::new(),
        }
    }

    /// Dimension of Krylov subspace
    pub fn dim(&self) -> usize {
        self.ortho.len()
    }

    /// Iterate until convergent
    pub fn complete(mut self) -> (Q<A>, H<A>) {
        for _ in &mut self {} // execute iteration until convergent
        let q = self.ortho.get_q();
        let n = self.h.len();
        let mut h = Array2::zeros((n, n).f());
        for (i, hc) in self.h.iter().enumerate() {
            let m = std::cmp::min(n, i + 2);
            for j in 0..m {
                h[(j, i)] = hc[j];
            }
        }
        (q, h)
    }
}

impl<A, S, F, Ortho> Iterator for Arnoldi<A, S, F, Ortho>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
    F: LinearOperator<Elem = A>,
    Ortho: Orthogonalizer<Elem = A>,
{
    type Item = Array1<A>;

    fn next(&mut self) -> Option<Self::Item> {
        self.a.apply_mut(&mut self.v);
        let result = self.ortho.div_append(&mut self.v);
        let norm = self.v.norm_l2();
        azip!((v in &mut self.v) *v = v.div_real(norm));
        match result {
            AppendResult::Added(coef) => {
                self.h.push(coef.clone());
                Some(coef)
            }
            AppendResult::Dependent(coef) => {
                self.h.push(coef);
                None
            }
        }
    }
}

/// Utility to execute Arnoldi iteration with Householder reflection
pub fn arnoldi_householder<A, S>(a: impl LinearOperator<Elem = A>, v: ArrayBase<S, Ix1>, tol: A::Real) -> (Q<A>, H<A>)
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
{
    let householder = Householder::new(v.len(), tol);
    Arnoldi::new(a, v, householder).complete()
}

/// Utility to execute Arnoldi iteration with modified Gram-Schmit orthogonalizer
pub fn arnoldi_mgs<A, S>(a: impl LinearOperator<Elem = A>, v: ArrayBase<S, Ix1>, tol: A::Real) -> (Q<A>, H<A>)
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
{
    let mgs = MGS::new(v.len(), tol);
    Arnoldi::new(a, v, mgs).complete()
}
