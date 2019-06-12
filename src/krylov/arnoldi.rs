use super::Orthogonalizer;
use crate::types::*;
use ndarray::*;

pub struct Arnoldi<S, F, Ortho>
where
    S: DataMut,
    F: Fn(&mut ArrayBase<S, Ix1>),
    Ortho: Orthogonalizer,
{
    a: F,
    v: ArrayBase<S, Ix1>,
    ortho: Ortho,
}

impl<S, F, Ortho> Arnoldi<S, F, Ortho>
where
    S: DataMut,
    F: Fn(&mut ArrayBase<S, Ix1>),
    Ortho: Orthogonalizer,
{
    pub fn new(a: F, v: ArrayBase<S, Ix1>, ortho: Ortho) -> Self {
        Arnoldi { a, v, ortho }
    }
}

impl<A, S, F, Ortho> Iterator for Arnoldi<S, F, Ortho>
where
    A: Scalar,
    S: DataMut<Elem = A>,
    F: Fn(&mut ArrayBase<S, Ix1>),
    Ortho: Orthogonalizer<Elem = A>,
{
    type Item = (Array2<A>, Array2<A>);

    fn next(&mut self) -> Option<Self::Item> {
        (self.a)(&mut self.v);
        let coef = self.ortho.decompose(&mut self.v);
        unimplemented!()
    }
}
