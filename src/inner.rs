use crate::types::*;
use ndarray::*;

/// Inner Product
///
/// Differenct from `Dot` trait, this take complex conjugate of `self` elements
///
pub trait InnerProduct {
    type Elem: Scalar;

    /// Inner product `(self.conjugate, rhs)
    fn inner<S>(&self, rhs: &ArrayBase<S, Ix1>) -> Self::Elem
    where
        S: Data<Elem = Self::Elem>;
}

impl<A, S> InnerProduct for ArrayBase<S, Ix1>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    type Elem = A;
    fn inner<St: Data<Elem = A>>(&self, rhs: &ArrayBase<St, Ix1>) -> A {
        assert_eq!(self.len(), rhs.len());
        Zip::from(self)
            .and(rhs)
            .fold_while(A::zero(), |acc, s, r| {
                FoldWhile::Continue(acc + s.conj() * *r)
            })
            .into_inner()
    }
}
