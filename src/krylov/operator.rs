//! Linear operator algebra

use ndarray::*;

pub trait LinearOperator {
    /// Apply operator out-place
    fn apply<S: Data>(&self, a: &ArrayBase<S, Ix1>) -> Array1<S::Elem> {
        self.apply_tensor(a, 0)
    }
    /// Apply operator in-place
    fn apply_mut<S: DataMut>(&self, a: &mut ArrayBase<S, Ix1>) {
        self.apply_tensor_mut(a, 0)
    }
    /// Apply operator with move
    fn apply_into<S: DataOwned + DataMut>(&self, a: ArrayBase<S, Ix1>) -> ArrayBase<S, Ix1> {
        self.apply_tensor_into(a, 0)
    }

    /// Apply operator to matrix out-place
    fn apply2<S: Data>(&self, a: &ArrayBase<S, Ix2>) -> Array2<S::Elem> {
        self.apply_tensor(a, 0)
    }
    /// Apply operator to matrix in-place
    fn apply2_mut<S: DataMut>(&self, a: &mut ArrayBase<S, Ix2>) {
        self.apply_tensor_mut(a, 0)
    }
    /// Apply operator to matrix with move
    fn apply2_into<S: DataOwned + DataMut>(&self, a: ArrayBase<S, Ix2>) -> ArrayBase<S, Ix2> {
        self.apply_tensor_into(a, 0)
    }

    /// Apply operator to the n-th index of the tensor (out-place)
    fn apply_tensor<S, D>(&self, a: &ArrayBase<S, D>, n: usize) -> Array<S::Elem, D>
    where
        S: Data,
        D: Dimension;
    /// Apply operator to the n-th index of the tensor (in-place)
    fn apply_tensor_mut<S, D>(&self, a: &mut ArrayBase<S, D>, n: usize)
    where
        S: DataMut,
        D: Dimension;
    /// Apply operator to the n-th index of the tensor (with move)
    fn apply_tensor_into<S, D>(&self, a: ArrayBase<S, D>, n: usize) -> ArrayBase<S, D>
    where
        S: DataOwned + DataMut,
        D: Dimension;
}
