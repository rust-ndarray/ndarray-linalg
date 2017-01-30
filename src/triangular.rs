
use ndarray::{Data, Ix1, Ix2, Array, RcArray, NdFloat, ArrayBase, DataMut};

use matrix::{Matrix, MFloat};
use square::SquareMatrix;
use error::LinalgError;
use solve::ImplSolve;

pub trait TriangularMatrix<Rhs>: Matrix + SquareMatrix {
    type Output;
    /// solve a triangular system with upper triangular matrix
    fn solve_upper(&self, &Rhs) -> Result<Self::Output, LinalgError>;
    /// solve a triangular system with lower triangular matrix
    fn solve_lower(&self, &Rhs) -> Result<Self::Output, LinalgError>;
}

impl<A, S> TriangularMatrix<ArrayBase<S, Ix1>> for Array<A, Ix2>
    where A: MFloat,
          S: Data<Elem = A>
{
    type Output = Array<A, Ix1>;

    fn solve_upper(&self, b: &ArrayBase<S, Ix1>) -> Result<Self::Output, LinalgError> {
        self.check_square()?;
        let (n, _) = self.size();
        let layout = self.layout()?;
        let a = self.as_slice_memory_order().unwrap();
        let x = ImplSolve::solve_triangle(layout, 'U' as u8, n, a, b.to_owned().into_raw_vec(), 1)?;
        Ok(Array::from_vec(x))
    }
    fn solve_lower(&self, b: &ArrayBase<S, Ix1>) -> Result<Self::Output, LinalgError> {
        self.check_square()?;
        let (n, _) = self.size();
        let layout = self.layout()?;
        let a = self.as_slice_memory_order().unwrap();
        let x = ImplSolve::solve_triangle(layout, 'L' as u8, n, a, b.to_owned().into_raw_vec(), 1)?;
        Ok(Array::from_vec(x))
    }
}

impl<A, S> TriangularMatrix<ArrayBase<S, Ix1>> for RcArray<A, Ix2>
    where A: MFloat,
          S: Data<Elem = A>
{
    type Output = RcArray<A, Ix1>;

    fn solve_upper(&self, b: &ArrayBase<S, Ix1>) -> Result<Self::Output, LinalgError> {
        // XXX unnecessary clone
        let x = self.to_owned().solve_upper(&b)?;
        Ok(x.into_shared())
    }
    fn solve_lower(&self, b: &ArrayBase<S, Ix1>) -> Result<Self::Output, LinalgError> {
        // XXX unnecessary clone
        let x = self.to_owned().solve_lower(&b)?;
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
