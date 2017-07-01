//! Linear Operator

use ndarray::*;

use super::convert::*;

pub trait Operator<A, S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    fn op(&self, &ArrayBase<S, D>) -> Array<A, D>;
}

pub trait OperatorInto<S, D>
where
    S: DataMut,
    D: Dimension,
{
    fn op_into(&self, ArrayBase<S, D>) -> ArrayBase<S, D>;
}

pub trait OperatorMut<S, D>
where
    S: DataMut,
    D: Dimension,
{
    fn op_mut<'a>(&self, &'a mut ArrayBase<S, D>) -> &'a mut ArrayBase<S, D>;
}

impl<T, A, S, D> Operator<A, S, D> for T
where
    A: LinalgScalar,
    S: Data<Elem = A>,
    D: Dimension,
    T: linalg::Dot<ArrayBase<S, D>, Output = Array<A, D>>,
{
    fn op(&self, rhs: &ArrayBase<S, D>) -> Array<A, D> {
        self.dot(rhs)
    }
}
