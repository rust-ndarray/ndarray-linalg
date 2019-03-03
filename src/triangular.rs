//! Methods for triangular matrices

use ndarray::*;
use num_traits::Zero;

use super::convert::*;
use super::error::*;
use super::lapack_traits::*;
use super::layout::*;
use super::types::*;

pub use super::lapack_traits::Diag;

/// solve a triangular system with upper triangular matrix
pub trait SolveTriangular<A, S, D>
where
    A: Scalar,
    S: Data<Elem = A>,
    D: Dimension,
{
    fn solve_triangular(&self, uplo: UPLO, diag: Diag, b: &ArrayBase<S, D>) -> Result<Array<A, D>>;
}

/// solve a triangular system with upper triangular matrix
pub trait SolveTriangularInto<S, D>
where
    S: DataMut,
    D: Dimension,
{
    fn solve_triangular_into(&self, uplo: UPLO, diag: Diag, b: ArrayBase<S, D>) -> Result<ArrayBase<S, D>>;
}

/// solve a triangular system with upper triangular matrix
pub trait SolveTriangularInplace<S, D>
where
    S: DataMut,
    D: Dimension,
{
    fn solve_triangular_inplace<'a>(
        &self,
        uplo: UPLO,
        diag: Diag,
        b: &'a mut ArrayBase<S, D>,
    ) -> Result<&'a mut ArrayBase<S, D>>;
}

impl<A, Si, So> SolveTriangularInto<So, Ix2> for ArrayBase<Si, Ix2>
where
    A: Scalar,
    Si: Data<Elem = A>,
    So: DataMut<Elem = A> + DataOwned,
{
    fn solve_triangular_into(&self, uplo: UPLO, diag: Diag, mut b: ArrayBase<So, Ix2>) -> Result<ArrayBase<So, Ix2>> {
        self.solve_triangular_inplace(uplo, diag, &mut b)?;
        Ok(b)
    }
}

impl<A, Si, So> SolveTriangularInplace<So, Ix2> for ArrayBase<Si, Ix2>
where
    A: Scalar,
    Si: Data<Elem = A>,
    So: DataMut<Elem = A> + DataOwned,
{
    fn solve_triangular_inplace<'a>(
        &self,
        uplo: UPLO,
        diag: Diag,
        b: &'a mut ArrayBase<So, Ix2>,
    ) -> Result<&'a mut ArrayBase<So, Ix2>> {
        let la = self.layout()?;
        let a_ = self.as_allocated()?;
        let lb = b.layout()?;
        if !la.same_order(&lb) {
            transpose_data(b)?;
        }
        let lb = b.layout()?;
        unsafe { A::solve_triangular(la, lb, uplo, diag, a_, b.as_allocated_mut()?)? };
        Ok(b)
    }
}

impl<A, Si, So> SolveTriangular<A, So, Ix2> for ArrayBase<Si, Ix2>
where
    A: Scalar,
    Si: Data<Elem = A>,
    So: DataMut<Elem = A> + DataOwned,
{
    fn solve_triangular(&self, uplo: UPLO, diag: Diag, b: &ArrayBase<So, Ix2>) -> Result<Array2<A>> {
        let b = replicate(b);
        self.solve_triangular_into(uplo, diag, b)
    }
}

impl<A, Si, So> SolveTriangularInto<So, Ix1> for ArrayBase<Si, Ix2>
where
    A: Scalar,
    Si: Data<Elem = A>,
    So: DataMut<Elem = A> + DataOwned,
{
    fn solve_triangular_into(&self, uplo: UPLO, diag: Diag, b: ArrayBase<So, Ix1>) -> Result<ArrayBase<So, Ix1>> {
        let b = into_col(b);
        let b = self.solve_triangular_into(uplo, diag, b)?;
        Ok(flatten(b))
    }
}

impl<A, Si, So> SolveTriangular<A, So, Ix1> for ArrayBase<Si, Ix2>
where
    A: Scalar,
    Si: Data<Elem = A>,
    So: DataMut<Elem = A> + DataOwned,
{
    fn solve_triangular(&self, uplo: UPLO, diag: Diag, b: &ArrayBase<So, Ix1>) -> Result<Array1<A>> {
        let b = b.to_owned();
        self.solve_triangular_into(uplo, diag, b)
    }
}

pub trait IntoTriangular<T> {
    fn into_triangular(self, uplo: UPLO) -> T;
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
