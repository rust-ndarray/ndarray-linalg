//! Vector as a Diagonal matrix

use ndarray::*;

use super::operator::*;
use super::types::*;

/// Vector as a Diagonal matrix
pub struct Diagonal<S: Data> {
    diag: ArrayBase<S, Ix1>,
}

pub trait IntoDiagonal<S: Data> {
    fn into_diagonal(self) -> Diagonal<S>;
}

pub trait AsDiagonal<A> {
    fn as_diagonal(&self) -> Diagonal<ViewRepr<&A>>;
}

impl<S: Data> IntoDiagonal<S> for ArrayBase<S, Ix1> {
    fn into_diagonal(self) -> Diagonal<S> {
        Diagonal { diag: self }
    }
}

impl<A, S: Data<Elem = A>> AsDiagonal<A> for ArrayBase<S, Ix1> {
    fn as_diagonal(&self) -> Diagonal<ViewRepr<&A>> {
        Diagonal { diag: self.view() }
    }
}

impl<A, Sa> LinearOperator for Diagonal<Sa>
where
    A: Scalar,
    Sa: Data<Elem = A>,
{
    type Elem = A;

    fn apply_mut<S>(&self, a: &mut ArrayBase<S, Ix1>)
    where
        S: DataMut<Elem = A>,
    {
        for (val, d) in a.iter_mut().zip(self.diag.iter()) {
            *val *= *d;
        }
    }
}
