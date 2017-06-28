//! Methods for triangular matrices

use ndarray::*;
use num_traits::Zero;

use super::error::*;
use super::lapack_traits::*;
use super::layout::*;

pub use super::lapack_traits::Diag;

/// solve a triangular system with upper triangular matrix
pub trait SolveTriangular<Rhs> {
    type Output;
    fn solve_triangular(&self, UPLO, Diag, Rhs) -> Result<Self::Output>;
}

impl<A, Si, So> SolveTriangular<ArrayBase<So, Ix2>> for ArrayBase<Si, Ix2>
where
    A: LapackScalar + Copy,
    Si: Data<Elem = A>,
    So: DataMut<Elem = A> + DataOwned,
{
    type Output = ArrayBase<So, Ix2>;

    fn solve_triangular(&self, uplo: UPLO, diag: Diag, mut b: ArrayBase<So, Ix2>) -> Result<Self::Output> {
        self.solve_triangular(uplo, diag, &mut b)?;
        Ok(b)
    }
}

impl<'a, A, Si, So> SolveTriangular<&'a mut ArrayBase<So, Ix2>> for ArrayBase<Si, Ix2>
where
    A: LapackScalar + Copy,
    Si: Data<Elem = A>,
    So: DataMut<Elem = A> + DataOwned,
{
    type Output = &'a mut ArrayBase<So, Ix2>;

    fn solve_triangular(&self, uplo: UPLO, diag: Diag, mut b: &'a mut ArrayBase<So, Ix2>) -> Result<Self::Output> {
        let la = self.layout()?;
        let a_ = self.as_allocated()?;
        let lb = b.layout()?;
        if !la.same_order(&lb) {
            data_transpose(b)?;
        }
        let lb = b.layout()?;
        A::solve_triangular(la, lb, uplo, diag, a_, b.as_allocated_mut()?)?;
        Ok(b)
    }
}

impl<'a, A, Si, So> SolveTriangular<&'a ArrayBase<So, Ix2>> for ArrayBase<Si, Ix2>
where
    A: LapackScalar + Copy,
    Si: Data<Elem = A>,
    So: DataMut<Elem = A> + DataOwned,
{
    type Output = ArrayBase<So, Ix2>;

    fn solve_triangular(&self, uplo: UPLO, diag: Diag, b: &'a ArrayBase<So, Ix2>) -> Result<Self::Output> {
        let b = replicate(b);
        self.solve_triangular(uplo, diag, b)
    }
}

impl<A, Si, So> SolveTriangular<ArrayBase<So, Ix1>> for ArrayBase<Si, Ix2>
where
    A: LapackScalar + Copy,
    Si: Data<Elem = A>,
    So: DataMut<Elem = A> + DataOwned,
{
    type Output = ArrayBase<So, Ix1>;

    fn solve_triangular(&self, uplo: UPLO, diag: Diag, b: ArrayBase<So, Ix1>) -> Result<Self::Output> {
        let b = into_col(b);
        let b = self.solve_triangular(uplo, diag, b)?;
        Ok(flatten(b))
    }
}

impl<'a, A, Si, So> SolveTriangular<&'a ArrayBase<So, Ix1>> for ArrayBase<Si, Ix2>
where
    A: LapackScalar + Copy,
    Si: Data<Elem = A>,
    So: DataMut<Elem = A> + DataOwned,
{
    type Output = ArrayBase<So, Ix1>;

    fn solve_triangular(&self, uplo: UPLO, diag: Diag, b: &'a ArrayBase<So, Ix1>) -> Result<Self::Output> {
        let b = replicate(b);
        self.solve_triangular(uplo, diag, b)
    }
}

pub trait IntoTriangular<T> {
    fn into_triangular(self, UPLO) -> T;
}

impl<'a, A, S> IntoTriangular<&'a mut ArrayBase<S, Ix2>> for &'a mut ArrayBase<S, Ix2>
where
    A: Zero,
    S: DataMut<Elem = A>,
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
where
    A: Zero,
    S: DataMut<Elem = A>,
{
    fn into_triangular(mut self, uplo: UPLO) -> ArrayBase<S, Ix2> {
        (&mut self).into_triangular(uplo);
        self
    }
}
