//! Linear operator algebra

use crate::generate::hstack;
use crate::types::*;
use ndarray::*;

pub trait LinearOperator {
    /// Apply operator out-place
    fn apply<S>(&self, a: &ArrayBase<S, Ix1>) -> Array1<S::Elem>
    where
        S: Data,
        S::Elem: Scalar,
    {
        let mut a = a.to_owned();
        self.apply_mut(&mut a);
        a
    }
    /// Apply operator in-place
    fn apply_mut<S>(&self, a: &mut ArrayBase<S, Ix1>)
    where
        S: DataMut,
        S::Elem: Scalar;
    /// Apply operator with move
    fn apply_into<S>(&self, mut a: ArrayBase<S, Ix1>) -> ArrayBase<S, Ix1>
    where
        S: DataOwned + DataMut,
        S::Elem: Scalar,
    {
        self.apply_mut(&mut a);
        a
    }

    /// Apply operator to matrix out-place
    fn apply2<S>(&self, a: &ArrayBase<S, Ix2>) -> Array2<S::Elem>
    where
        S: Data,
        S::Elem: Scalar,
    {
        let cols: Vec<_> = a.axis_iter(Axis(0)).map(|col| self.apply(&col)).collect();
        hstack(&cols).unwrap()
    }
    /// Apply operator to matrix in-place
    fn apply2_mut<S>(&self, a: &mut ArrayBase<S, Ix2>)
    where
        S: DataMut,
        S::Elem: Scalar,
    {
        for mut col in a.axis_iter_mut(Axis(0)) {
            self.apply_mut(&mut col)
        }
    }
    /// Apply operator to matrix with move
    fn apply2_into<S>(&self, mut a: ArrayBase<S, Ix2>) -> ArrayBase<S, Ix2>
    where
        S: DataOwned + DataMut,
        S::Elem: Scalar,
    {
        self.apply2_mut(&mut a);
        a
    }
}
