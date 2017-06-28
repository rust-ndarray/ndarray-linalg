//! Linear Operator

use ndarray::*;

use super::convert::*;

/// General operator trait. It extends `ndarray::linalg::Dot`
pub trait Operator<RHS, Output> {
    fn op(&self, RHS) -> Output;
}

impl<'a, A, S, Si, So> Operator<&'a ArrayBase<Si, Ix1>, ArrayBase<So, Ix1>> for ArrayBase<S, Ix2>
where
    A: LinalgScalar,
    S: Data<Elem = A>,
    Si: Data<Elem = A>,
    So: DataOwned<Elem = A>,
{
    fn op(&self, a: &'a ArrayBase<Si, Ix1>) -> ArrayBase<So, Ix1> {
        generalize(self.dot(a))
    }
}

impl<'a, A, S, Si, So> Operator<&'a ArrayBase<Si, Ix2>, ArrayBase<So, Ix2>> for ArrayBase<S, Ix2>
where
    A: LinalgScalar,
    S: Data<Elem = A>,
    Si: Data<Elem = A>,
    So: DataOwned<Elem = A>,
{
    fn op(&self, a: &'a ArrayBase<Si, Ix2>) -> ArrayBase<So, Ix2> {
        generalize(self.dot(a))
    }
}
