
use ndarray::{Ix2, Array, RcArray, NdFloat, ArrayBase, DataMut};

use matrix::{Matrix, MFloat};
use square::SquareMatrix;
use error::LinalgError;
use solve::ImplSolve;

pub trait TriangularMatrix: Matrix + SquareMatrix {
    /// solve a triangular system with upper triangular matrix
    fn solve_upper(&self, Self::Vector) -> Result<Self::Vector, LinalgError>;
    /// solve a triangular system with lower triangular matrix
    fn solve_lower(&self, Self::Vector) -> Result<Self::Vector, LinalgError>;
}

impl<A: MFloat> TriangularMatrix for Array<A, Ix2> {
    fn solve_upper(&self, b: Self::Vector) -> Result<Self::Vector, LinalgError> {
        self.check_square()?;
        let (n, _) = self.size();
        let layout = self.layout()?;
        let a = self.as_slice_memory_order().unwrap();
        let x = ImplSolve::solve_triangle(layout, 'U' as u8, n, a, b.into_raw_vec())?;
        Ok(Array::from_vec(x))
    }
    fn solve_lower(&self, b: Self::Vector) -> Result<Self::Vector, LinalgError> {
        self.check_square()?;
        let (n, _) = self.size();
        let layout = self.layout()?;
        let a = self.as_slice_memory_order().unwrap();
        let x = ImplSolve::solve_triangle(layout, 'L' as u8, n, a, b.into_raw_vec())?;
        Ok(Array::from_vec(x))
    }
}

impl<A: MFloat> TriangularMatrix for RcArray<A, Ix2> {
    fn solve_upper(&self, b: Self::Vector) -> Result<Self::Vector, LinalgError> {
        let x = self.to_owned().solve_upper(b.to_owned())?;
        Ok(x.into_shared())
    }
    fn solve_lower(&self, b: Self::Vector) -> Result<Self::Vector, LinalgError> {
        let x = self.to_owned().solve_lower(b.to_owned())?;
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
