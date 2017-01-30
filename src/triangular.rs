
use ndarray::{Ix1, Ix2, Array, RcArray, NdFloat, ArrayBase, DataMut};

use matrix::{Matrix, MFloat};
use square::SquareMatrix;
use error::LinalgError;
use solve::ImplSolve;

pub trait SolveTriangular<Rhs>: Matrix + SquareMatrix {
    type Output;
    /// solve a triangular system with upper triangular matrix
    fn solve_upper(&self, Rhs) -> Result<Self::Output, LinalgError>;
    /// solve a triangular system with lower triangular matrix
    fn solve_lower(&self, Rhs) -> Result<Self::Output, LinalgError>;
}

impl<A: MFloat> SolveTriangular<Array<A, Ix1>> for Array<A, Ix2> {
    type Output = Array<A, Ix1>;
    fn solve_upper(&self, b: Array<A, Ix1>) -> Result<Self::Output, LinalgError> {
        let n = self.square_size()?;
        let layout = self.layout()?;
        let a = self.as_slice_memory_order().unwrap();
        let x = ImplSolve::solve_triangle(layout, 'U' as u8, n, a, b.into_raw_vec(), 1)?;
        Ok(Array::from_vec(x))
    }
    fn solve_lower(&self, b: Array<A, Ix1>) -> Result<Self::Output, LinalgError> {
        let n = self.square_size()?;
        let layout = self.layout()?;
        let a = self.as_slice_memory_order().unwrap();
        let x = ImplSolve::solve_triangle(layout, 'L' as u8, n, a, b.into_raw_vec(), 1)?;
        Ok(Array::from_vec(x))
    }
}

impl<A: MFloat> SolveTriangular<RcArray<A, Ix1>> for RcArray<A, Ix2> {
    type Output = RcArray<A, Ix1>;
    fn solve_upper(&self, b: RcArray<A, Ix1>) -> Result<Self::Output, LinalgError> {
        // XXX unnecessary clone
        let x = self.to_owned().solve_upper(b.into_owned())?;
        Ok(x.into_shared())
    }
    fn solve_lower(&self, b: RcArray<A, Ix1>) -> Result<Self::Output, LinalgError> {
        // XXX unnecessary clone
        let x = self.to_owned().solve_lower(b.into_owned())?;
        Ok(x.into_shared())
    }
}

pub fn drop_upper<A: NdFloat, S>(mut a: ArrayBase<S, Ix2>) -> ArrayBase<S, Ix2>
    where S: DataMut<Elem = A>
{
    for ((i, j), val) in a.indexed_iter_mut() {
        if i < j {
            *val = A::zero();
        }
    }
    a
}

pub fn drop_lower<A: NdFloat, S>(mut a: ArrayBase<S, Ix2>) -> ArrayBase<S, Ix2>
    where S: DataMut<Elem = A>
{
    for ((i, j), val) in a.indexed_iter_mut() {
        if i > j {
            *val = A::zero();
        }
    }
    a
}
