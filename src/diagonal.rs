//! Vector as a Diagonal matrix

use ndarray::*;

use super::convert::*;
use super::operator::*;

/// Vector as a Diagonal matrix
pub struct Diagonal<S: Data> {
    diag: ArrayBase<S, Ix1>,
}

pub trait IntoDiagonal<S: Data> {
    fn into_diagonal(self) -> Diagonal<S>;
}

pub trait AsDiagonal<A> {
    fn as_diagonal<'a>(&'a self) -> Diagonal<ViewRepr<&'a A>>;
}

impl<S: Data> IntoDiagonal<S> for ArrayBase<S, Ix1> {
    fn into_diagonal(self) -> Diagonal<S> {
        Diagonal { diag: self }
    }
}

impl<A, S: Data<Elem = A>> AsDiagonal<A> for ArrayBase<S, Ix1> {
    fn as_diagonal<'a>(&'a self) -> Diagonal<ViewRepr<&'a A>> {
        Diagonal { diag: self.view() }
    }
}

impl<A, S, Sr> OperatorMut<Sr, Ix1> for Diagonal<S>
where
    A: LinalgScalar,
    S: Data<Elem = A>,
    Sr: DataMut<Elem = A>,
{
    fn op_mut<'a>(&self, mut a: &'a mut ArrayBase<Sr, Ix1>) -> &'a mut ArrayBase<Sr, Ix1> {
        for (val, d) in a.iter_mut().zip(self.diag.iter()) {
            *val = *val * *d;
        }
        a
    }
}

impl<A, S, Sr> Operator<A, Sr, Ix1> for Diagonal<S>
where
    A: LinalgScalar,
    S: Data<Elem = A>,
    Sr: Data<Elem = A>,
{
    fn op(&self, a: &ArrayBase<Sr, Ix1>) -> Array1<A> {
        let mut a = replicate(a);
        self.op_mut(&mut a);
        a
    }
}

impl<A, S, Sr> OperatorInto<Sr, Ix1> for Diagonal<S>
where
    A: LinalgScalar,
    S: Data<Elem = A>,
    Sr: DataOwned<Elem = A> + DataMut,
{
    fn op_into(&self, mut a: ArrayBase<Sr, Ix1>) -> ArrayBase<Sr, Ix1> {
        self.op(&mut a);
        a
    }
}

impl<A, S, Sr> OperatorMut<Sr, Ix2> for Diagonal<S>
where
    A: LinalgScalar,
    S: Data<Elem = A>,
    Sr: DataMut<Elem = A>,
{
    fn op_mut<'a>(&self, mut a: &'a mut ArrayBase<Sr, Ix2>) -> &'a mut ArrayBase<Sr, Ix2> {
        let ref d = self.diag;
        for ((i, _), val) in a.indexed_iter_mut() {
            *val = *val * d[i];
        }
        a
    }
}

impl<A, S, Sr> Operator<A, Sr, Ix2> for Diagonal<S>
where
    A: LinalgScalar,
    S: Data<Elem = A>,
    Sr: Data<Elem = A>,
{
    fn op(&self, a: &ArrayBase<Sr, Ix2>) -> Array2<A> {
        let mut a = replicate(a);
        self.op_mut(&mut a);
        a
    }
}

impl<A, S, Sr> OperatorInto<Sr, Ix2> for Diagonal<S>
where
    A: LinalgScalar,
    S: Data<Elem = A>,
    Sr: DataOwned<Elem = A> + DataMut,
{
    fn op_into(&self, mut a: ArrayBase<Sr, Ix2>) -> ArrayBase<Sr, Ix2> {
        self.op(&mut a);
        a
    }
}
