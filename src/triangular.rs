//! Define methods for triangular matrices

use ndarray::*;
use num_traits::Zero;

use super::layout::*;
use super::error::*;
use super::impl2::*;

pub use super::impl2::Diag;

/// solve a triangular system with upper triangular matrix
pub trait SolveTriangular<Rhs> {
    type Output;
    fn solve_triangular(&self, UPLO, Diag, Rhs) -> Result<Self::Output>;
}

impl<A, Si, So, D> SolveTriangular<ArrayBase<So, D>> for ArrayBase<Si, Ix2>
    where A: LapackScalar,
          Si: Data<Elem = A>,
          So: DataMut<Elem = A>,
          D: Dimension,
          ArrayBase<So, D>: AllocatedArrayMut<Elem = A>
{
    type Output = ArrayBase<So, D>;

    fn solve_triangular(&self, uplo: UPLO, diag: Diag, mut b: ArrayBase<So, D>) -> Result<Self::Output> {
        self.solve_triangular(uplo, diag, &mut b)?;
        Ok(b)
    }
}

impl<'a, A, Si, So, D> SolveTriangular<&'a mut ArrayBase<So, D>> for ArrayBase<Si, Ix2>
    where A: LapackScalar,
          Si: Data<Elem = A>,
          So: DataMut<Elem = A>,
          D: Dimension,
          ArrayBase<So, D>: AllocatedArrayMut<Elem = A>
{
    type Output = &'a mut ArrayBase<So, D>;

    fn solve_triangular(&self, uplo: UPLO, diag: Diag, mut b: &'a mut ArrayBase<So, D>) -> Result<Self::Output> {
        let la = self.layout()?;
        let lb = b.layout()?;
        let a_ = self.as_allocated()?;
        A::solve_triangular(la, lb, uplo, diag, a_, b.as_allocated_mut()?)?;
        Ok(b)
    }
}

impl<'a, A, Si, So, D> SolveTriangular<&'a ArrayBase<So, D>> for ArrayBase<Si, Ix2>
    where A: LapackScalar + Copy,
          Si: Data<Elem = A>,
          So: DataMut<Elem = A> + DataOwned,
          D: Dimension,
          ArrayBase<So, D>: AllocatedArrayMut<Elem = A>
{
    type Output = ArrayBase<So, D>;

    fn solve_triangular(&self, uplo: UPLO, diag: Diag, b: &'a ArrayBase<So, D>) -> Result<Self::Output> {
        let b = replicate(b);
        self.solve_triangular(uplo, diag, b)
    }
}

pub trait IntoTriangular<T> {
    fn into_triangular(self, UPLO) -> T;
}

impl<'a, A, S> IntoTriangular<&'a mut ArrayBase<S, Ix2>> for &'a mut ArrayBase<S, Ix2>
    where A: Zero,
          S: DataMut<Elem = A>
{
    fn into_triangular(self, uplo: UPLO) -> &'a mut ArrayBase<S, Ix2> {
        match uplo {
            UPLO::Upper => {
                for ((i, j), val) in self.indexed_iter_mut() {
                    if i > j {
                        *val = A::zero();
                    }
                }
            }
            UPLO::Lower => {
                for ((i, j), val) in self.indexed_iter_mut() {
                    if i < j {
                        *val = A::zero();
                    }
                }
            }
        }
        self
    }
}

impl<A, S> IntoTriangular<ArrayBase<S, Ix2>> for ArrayBase<S, Ix2>
    where A: Zero,
          S: DataMut<Elem = A>
{
    fn into_triangular(mut self, uplo: UPLO) -> ArrayBase<S, Ix2> {
        (&mut self).into_triangular(uplo);
        self
    }
}

pub fn drop_upper<A: Zero, S>(a: ArrayBase<S, Ix2>) -> ArrayBase<S, Ix2>
    where S: DataMut<Elem = A>
{
    a.into_triangular(UPLO::Lower)
}

pub fn drop_lower<A: Zero, S>(a: ArrayBase<S, Ix2>) -> ArrayBase<S, Ix2>
    where S: DataMut<Elem = A>
{
    a.into_triangular(UPLO::Upper)
}
