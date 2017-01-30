
use ndarray::*;
use super::matrix::{Matrix, MFloat};
use super::square::SquareMatrix;
use super::error::LinalgError;
use super::solve::ImplSolve;
use super::util::hstack;

pub trait SolveTriangular<Rhs>: Matrix + SquareMatrix {
    type Output;
    /// solve a triangular system with upper triangular matrix
    fn solve_upper(&self, Rhs) -> Result<Self::Output, LinalgError>;
    /// solve a triangular system with lower triangular matrix
    fn solve_lower(&self, Rhs) -> Result<Self::Output, LinalgError>;
}

impl<A, S1, S2> SolveTriangular<ArrayBase<S2, Ix1>> for ArrayBase<S1, Ix2>
    where A: MFloat,
          S1: Data<Elem = A>,
          S2: DataMut<Elem = A>,
          ArrayBase<S1, Ix2>: Matrix + SquareMatrix
{
    type Output = ArrayBase<S2, Ix1>;
    fn solve_upper(&self, mut b: ArrayBase<S2, Ix1>) -> Result<Self::Output, LinalgError> {
        let n = self.square_size()?;
        let layout = self.layout()?;
        let a = self.as_slice_memory_order().unwrap();
        {
            let b_ = b.as_slice_memory_order_mut().unwrap();
            ImplSolve::solve_triangle(layout, 'U' as u8, n, a, b_, 1)?;
        }
        Ok(b)
    }
    fn solve_lower(&self, mut b: ArrayBase<S2, Ix1>) -> Result<Self::Output, LinalgError> {
        let n = self.square_size()?;
        let layout = self.layout()?;
        let a = self.as_slice_memory_order().unwrap();
        {
            let b_ = b.as_slice_memory_order_mut().unwrap();
            ImplSolve::solve_triangle(layout, 'L' as u8, n, a, b_, 1)?;
        }
        Ok(b)
    }
}

impl<'a, S1, S2, A> SolveTriangular<&'a ArrayBase<S2, Ix2>> for ArrayBase<S1, Ix2>
    where A: MFloat,
          S1: Data<Elem = A>,
          S2: Data<Elem = A>,
          ArrayBase<S1, Ix2>: Matrix + SquareMatrix
{
    type Output = Array<A, Ix2>;
    fn solve_upper(&self, bs: &ArrayBase<S2, Ix2>) -> Result<Self::Output, LinalgError> {
        let mut xs = Vec::new();
        for b in bs.axis_iter(Axis(1)) {
            let x = self.solve_upper(b.to_owned())?;
            xs.push(x);
        }
        hstack(&xs).map_err(|e| e.into())
    }
    fn solve_lower(&self, bs: &ArrayBase<S2, Ix2>) -> Result<Self::Output, LinalgError> {
        let mut xs = Vec::new();
        for b in bs.axis_iter(Axis(1)) {
            let x = self.solve_lower(b.to_owned())?;
            xs.push(x);
        }
        hstack(&xs).map_err(|e| e.into())
    }
}

impl<A: MFloat> SolveTriangular<RcArray<A, Ix2>> for RcArray<A, Ix2> {
    type Output = RcArray<A, Ix2>;
    fn solve_upper(&self, b: RcArray<A, Ix2>) -> Result<Self::Output, LinalgError> {
        // XXX unnecessary clone
        let x = self.to_owned().solve_upper(&b)?;
        Ok(x.into_shared())
    }
    fn solve_lower(&self, b: RcArray<A, Ix2>) -> Result<Self::Output, LinalgError> {
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
