//! Linear Operator

use ndarray::*;

use super::types::*;

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

pub trait OperatorInplace<S, D>
where
    S: DataMut,
    D: Dimension,
{
    fn op_inplace<'a>(&self, &'a mut ArrayBase<S, D>) -> &'a mut ArrayBase<S, D>;
}

impl<T, A, S, D> Operator<A, S, D> for T
where
    A: Scalar,
    S: Data<Elem = A>,
    D: Dimension,
    T: linalg::Dot<ArrayBase<S, D>, Output = Array<A, D>>,
{
    fn op(&self, rhs: &ArrayBase<S, D>) -> Array<A, D> {
        self.dot(rhs)
    }
}

pub trait OperatorMulti<A, S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    fn op_multi(&self, &ArrayBase<S, D>) -> Array<A, D>;
}

impl<T, A, S, D> OperatorMulti<A, S, D> for T
where
    A: Scalar,
    S: DataMut<Elem = A>,
    D: Dimension + RemoveAxis,
    for<'a> T: OperatorInplace<ViewRepr<&'a mut A>, D::Smaller>,
{
    fn op_multi(&self, a: &ArrayBase<S, D>) -> Array<A, D> {
        let a = a.to_owned();
        self.op_multi_into(a)
    }
}

pub trait OperatorMultiInto<S, D>
where
    S: DataMut,
    D: Dimension,
{
    fn op_multi_into(&self, ArrayBase<S, D>) -> ArrayBase<S, D>;
}

impl<T, A, S, D> OperatorMultiInto<S, D> for T
where
    S: DataMut<Elem = A>,
    D: Dimension + RemoveAxis,
    for<'a> T: OperatorInplace<ViewRepr<&'a mut A>, D::Smaller>,
{
    fn op_multi_into(&self, mut a: ArrayBase<S, D>) -> ArrayBase<S, D> {
        self.op_multi_inplace(&mut a);
        a
    }
}

pub trait OperatorMultiInplace<S, D>
where
    S: DataMut,
    D: Dimension,
{
    fn op_multi_inplace<'a>(&self, &'a mut ArrayBase<S, D>) -> &'a mut ArrayBase<S, D>;
}

impl<T, A, S, D> OperatorMultiInplace<S, D> for T
where
    S: DataMut<Elem = A>,
    D: Dimension + RemoveAxis,
    for<'a> T: OperatorInplace<ViewRepr<&'a mut A>, D::Smaller>,
{
    fn op_multi_inplace<'a>(&self, mut a: &'a mut ArrayBase<S, D>) -> &'a mut ArrayBase<S, D> {
        let n = a.ndim();
        for mut col in a.axis_iter_mut(Axis(n - 1)) {
            self.op_inplace(&mut col);
        }
        a
    }
}
