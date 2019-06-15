use super::*;
use crate::norm::Norm;
use num_traits::One;
use std::iter::*;

/// Execute Arnodi iteration as Rust iterator
///
/// - [Arnoldi iteration - Wikipedia](https://en.wikipedia.org/wiki/Arnoldi_iteration)
///
pub struct Arnoldi<A, S, F, Ortho>
where
    A: Scalar,
    S: DataMut<Elem = A>,
    F: Fn(&mut ArrayBase<S, Ix1>),
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
    F: Fn(&mut ArrayBase<S, Ix1>),
    Ortho: Orthogonalizer<Elem = A>,
{
    pub fn new(a: F, mut v: ArrayBase<S, Ix1>, mut ortho: Ortho) -> Self {
        assert_eq!(ortho.len(), 0);
        assert!(ortho.tolerance() < One::one());
        // normalize before append because |v| may be smaller than ortho.tolerance()
        let norm = v.norm_l2();
        azip!(mut v(&mut v) in { *v = v.div_real(norm) });
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
    F: Fn(&mut ArrayBase<S, Ix1>),
    Ortho: Orthogonalizer<Elem = A>,
{
    type Item = Array1<A>;

    fn next(&mut self) -> Option<Self::Item> {
        (self.a)(&mut self.v);
        let result = self.ortho.div_append(&mut self.v);
        let norm = self.v.norm_l2();
        azip!(mut v(&mut self.v) in { *v = v.div_real(norm) });
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

/// Interpret a matrix as a linear operator
pub fn mul_mat<A, S1, S2>(a: ArrayBase<S1, Ix2>) -> impl Fn(&mut ArrayBase<S2, Ix1>)
where
    A: Scalar,
    S1: Data<Elem = A>,
    S2: DataMut<Elem = A>,
{
    let (n, m) = a.dim();
    assert_eq!(n, m, "Input matrix must be square");
    move |x| {
        assert_eq!(m, x.len(), "Input matrix and vector sizes mismatch");
        let ax = a.dot(x);
        azip!(mut x(x), ax in { *x = ax });
    }
}

/// Utility to execute Arnoldi iteration with Householder reflection
pub fn arnoldi_householder<A, S1, S2>(a: ArrayBase<S1, Ix2>, v: ArrayBase<S2, Ix1>, tol: A::Real) -> (Q<A>, H<A>)
where
    A: Scalar + Lapack,
    S1: Data<Elem = A>,
    S2: DataMut<Elem = A>,
{
    let householder = Householder::new(v.len(), tol);
    Arnoldi::new(mul_mat(a), v, householder).complete()
}

/// Utility to execute Arnoldi iteration with modified Gram-Schmit orthogonalizer
pub fn arnoldi_mgs<A, S1, S2>(a: ArrayBase<S1, Ix2>, v: ArrayBase<S2, Ix1>, tol: A::Real) -> (Q<A>, H<A>)
where
    A: Scalar + Lapack,
    S1: Data<Elem = A>,
    S2: DataMut<Elem = A>,
{
    let mgs = MGS::new(v.len(), tol);
    Arnoldi::new(mul_mat(a), v, mgs).complete()
}
