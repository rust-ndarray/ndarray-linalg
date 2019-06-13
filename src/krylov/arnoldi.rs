use super::*;
use crate::norm::Norm;
use num_traits::One;
use std::iter::Fuse;

pub struct Arnoldi<A, S, F, Ortho>
where
    A: Scalar,
    S: DataMut<Elem = A>,
    F: Fn(&mut ArrayBase<S, Ix1>),
    Ortho: Orthogonalizer<Elem = A>,
{
    a: F,
    v: ArrayBase<S, Ix1>,
    ortho: Ortho,
}

impl<A, S, F, Ortho> Arnoldi<A, S, F, Ortho>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
    F: Fn(&mut ArrayBase<S, Ix1>),
    Ortho: Orthogonalizer<Elem = A>,
{
    pub fn new(a: F, mut v: ArrayBase<S, Ix1>, mut ortho: Ortho) -> Fuse<Self> {
        assert_eq!(ortho.len(), 0);
        assert!(ortho.tolerance() < One::one());
        // normalize before append because |v| may be smaller than ortho.tolerance()
        let norm = v.norm_l2();
        azip!(mut v(&mut v) in { *v = v.div_real(norm) });
        ortho.append(v.view());
        Iterator::fuse(Arnoldi { a, v, ortho })
    }

    /// Dimension of Krylov subspace
    pub fn krylov_dimension(&self) -> usize {
        self.ortho.len()
    }
}

impl<A, S, F, Ortho> Iterator for Arnoldi<A, S, F, Ortho>
where
    A: Scalar,
    S: DataMut<Elem = A>,
    F: Fn(&mut ArrayBase<S, Ix1>),
    Ortho: Orthogonalizer<Elem = A>,
{
    type Item = Array1<A>;

    fn next(&mut self) -> Option<Self::Item> {
        (self.a)(&mut self.v);
        let result = self.ortho.div_append(&mut self.v);
        azip!(mut v(&mut self.v) in { *v = v.div_real(result.residual_norm()) });
        if result.is_dependent() {
            None
        } else {
            Some(result.into_coeff())
        }
    }
}

pub fn arnoldi_householder<A, S1, S2>(
    a: ArrayBase<S1, Ix2>,
    v: ArrayBase<S2, Ix1>,
    tol: A::Real,
) -> impl Iterator<Item = Array1<A>>
where
    A: Scalar + Lapack,
    S1: Data<Elem = A>,
    S2: DataMut<Elem = A>,
{
    let (n, m) = a.dim();
    assert_eq!(n, m, "Input matrix must be square");
    assert_eq!(m, v.len(), "Input matrix and vector sizes mismach");

    let householder = Householder::new(n, tol);
    Arnoldi::new(
        move |x| {
            let ax = a.dot(x);
            azip!(mut x(x), ax in { *x = ax });
        },
        v,
        householder,
    )
}

pub fn arnoldi_mgs<A, S1, S2>(
    a: ArrayBase<S1, Ix2>,
    v: ArrayBase<S2, Ix1>,
    tol: A::Real,
) -> impl Iterator<Item = Array1<A>>
where
    A: Scalar + Lapack,
    S1: Data<Elem = A>,
    S2: DataMut<Elem = A>,
{
    let (n, m) = a.dim();
    assert_eq!(n, m, "Input matrix must be square");
    assert_eq!(m, v.len(), "Input matrix and vector sizes mismach");

    let mgs = MGS::new(n, tol);
    Arnoldi::new(
        move |x| {
            let ax = a.dot(x);
            azip!(mut x(x), ax in { *x = ax });
        },
        v,
        mgs,
    )
}
