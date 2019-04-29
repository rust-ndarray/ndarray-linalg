use crate::types::*;
use ndarray::*;

pub trait Inner<S: Data> {
    /// Inner product `(self.conjugate, rhs)`
    fn inner(&self, rhs: &ArrayBase<S, Ix1>) -> S::Elem;
}

impl<A, S1, S2> Inner<S1> for ArrayBase<S2, Ix1>
where
    A: Scalar,
    S1: Data<Elem = A>,
    S2: Data<Elem = A>,
{
    fn inner(&self, rhs: &ArrayBase<S1, Ix1>) -> A {
        Zip::from(self)
            .and(rhs)
            .fold_while(A::zero(), |acc, s, r| FoldWhile::Continue(acc + s.conj() * *r))
            .into_inner()
    }
}
