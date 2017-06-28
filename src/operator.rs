use ndarray::*;

use super::convert::*;

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
