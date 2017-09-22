//! Cholesky decomposition of Hermitian (or real symmetric) positive definite matrices
//!
//! See the [Wikipedia page about Cholesky
//! decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition) for
//! more information.
//!
//! # Example
//!
//! Using the Cholesky decomposition of `A` for various operations, where `A`
//! is a Hermitian (or real symmetric) positive definite matrix:
//!
//! ```
//! #[macro_use]
//! extern crate ndarray;
//! extern crate ndarray_linalg;
//!
//! use ndarray::prelude::*;
//! use ndarray_linalg::{Cholesky, UPLO};
//! # fn main() {
//!
//! let a: Array2<f64> = array![
//!     [  4.,  12., -16.],
//!     [ 12.,  37., -43.],
//!     [-16., -43.,  98.]
//! ];
//! let chol_lower = a.cholesky(UPLO::Lower).unwrap();
//!
//! // Examine `L`
//! assert!(chol_lower.factor.all_close(&array![
//!     [ 2., 0., 0.],
//!     [ 6., 1., 0.],
//!     [-8., 5., 3.]
//! ], 1e-9));
//!
//! // Find the determinant of `A`
//! let det = chol_lower.det();
//! assert!((det - 36.).abs() < 1e-9);
//!
//! // Solve `A * x = b`
//! let b = array![4., 13., -11.];
//! let x = chol_lower.solve(&b).unwrap();
//! assert!(x.all_close(&array![-2., 1., 0.], 1e-9));
//! # }
//! ```

use ndarray::*;
use num_traits::Float;

use super::convert::*;
use super::error::*;
use super::layout::*;
use super::triangular::IntoTriangular;
use super::types::*;

pub use lapack_traits::UPLO;

/// Cholesky decomposition of Hermitian (or real symmetric) positive definite matrix
pub struct FactorizedCholesky<S>
where
    S: Data,
{
    /// `L` from the decomposition `A = L * L^H` or `U` from the decomposition
    /// `A = U^H * U`.
    pub factor: ArrayBase<S, Ix2>,
    /// If this is `UPLO::Lower`, then `self.factor` is `L`. If this is
    /// `UPLO::Upper`, then `self.factor` is `U`.
    pub uplo: UPLO,
}

impl<A, S> FactorizedCholesky<S>
where
    A: Scalar,
    S: DataMut<Elem = A>,
{
    /// Returns `L` from the Cholesky decomposition `A = L * L^H`.
    ///
    /// If `self.uplo == UPLO::Lower`, then no computations need to be
    /// performed; otherwise, the conjugate transpose of `self.factor` is
    /// calculated.
    pub fn into_lower(self) -> ArrayBase<S, Ix2> {
        match self.uplo {
            UPLO::Lower => self.factor,
            UPLO::Upper => self.factor.reversed_axes().mapv_into(|elem| elem.conj()),
        }
    }

    /// Returns `U` from the Cholesky decomposition `A = U^H * U`.
    ///
    /// If `self.uplo == UPLO::Upper`, then no computations need to be
    /// performed; otherwise, the conjugate transpose of `self.factor` is
    /// calculated.
    pub fn into_upper(self) -> ArrayBase<S, Ix2> {
        match self.uplo {
            UPLO::Lower => self.factor.reversed_axes().mapv_into(|elem| elem.conj()),
            UPLO::Upper => self.factor,
        }
    }

    /// Computes the inverse of the Cholesky-factored matrix.
    ///
    /// **Warning: The inverse is stored only in the triangular portion of the
    /// result matrix corresponding to `self.uplo`!** If you want the other
    /// triangular portion to be correct, you must fill it in yourself.
    pub fn into_inverse(mut self) -> Result<ArrayBase<S, Ix2>> {
        unsafe {
            A::inv_cholesky(
                self.factor.square_layout()?,
                self.uplo,
                self.factor.as_allocated_mut()?,
            )?
        };
        Ok(self.factor)
    }
}

impl<A, S> FactorizedCholesky<S>
where
    A: Absolute,
    S: Data<Elem = A>,
{
    /// Computes the natural log of the determinant of the Cholesky-factored
    /// matrix.
    pub fn ln_det(&self) -> <A as AssociatedReal>::Real {
        self.factor
            .diag()
            .iter()
            .map(|elem| elem.abs_sqr().ln())
            .sum()
    }

    /// Computes the determinant of the Cholesky-factored matrix.
    pub fn det(&self) -> <A as AssociatedReal>::Real {
        self.ln_det().exp()
    }
}

impl<A, S> FactorizedCholesky<S>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    /// Solves a system of linear equations `A * x = b`, where `self` is the
    /// Cholesky factorization of `A`, `b` is the argument, and `x` is the
    /// successful result.
    pub fn solve<Sb>(&self, b: &ArrayBase<Sb, Ix1>) -> Result<Array1<A>>
    where
        Sb: Data<Elem = A>,
    {
        let mut b = replicate(b);
        self.solve_mut(&mut b)?;
        Ok(b)
    }

    /// Solves a system of linear equations `A * x = b`, where `self` is the
    /// Cholesky factorization `A`, `b` is the argument, and `x` is the
    /// successful result.
    pub fn solve_into<Sb>(&self, mut b: ArrayBase<Sb, Ix1>) -> Result<ArrayBase<Sb, Ix1>>
    where
        Sb: DataMut<Elem = A>,
    {
        self.solve_mut(&mut b)?;
        Ok(b)
    }

    /// Solves a system of linear equations `A * x = b`, where `self` is the
    /// Cholesky factorization of `A`, `b` is the argument, and `x` is the
    /// successful result. The value of `x` is also assigned to the argument.
    pub fn solve_mut<'a, Sb>(&self, b: &'a mut ArrayBase<Sb, Ix1>) -> Result<&'a mut ArrayBase<Sb, Ix1>>
    where
        Sb: DataMut<Elem = A>,
    {
        unsafe {
            A::solve_cholesky(
                self.factor.square_layout()?,
                self.uplo,
                self.factor.as_allocated()?,
                b.as_slice_mut().unwrap(),
            )?
        };
        Ok(b)
    }
}

/// Cholesky decomposition of Hermitian (or real symmetric) positive definite matrix reference
pub trait Cholesky<S: Data> {
    /// Computes the Cholesky decomposition of the Hermitian (or real
    /// symmetric) positive definite matrix.
    ///
    /// If the argument is `UPLO::Upper`, then computes the decomposition `A =
    /// U^H * U` using the upper triangular portion of `A` and returns the
    /// factorization containing `U`. Otherwise, if the argument is
    /// `UPLO::Lower`, computes the decomposition `A = L * L^H` using the lower
    /// triangular portion of `A` and returns the factorization containing `L`.
    fn cholesky(&self, UPLO) -> Result<FactorizedCholesky<S>>;
}

/// Cholesky decomposition of Hermitian (or real symmetric) positive definite matrix
pub trait CholeskyInto<S: Data> {
    /// Computes the Cholesky decomposition of the Hermitian (or real
    /// symmetric) positive definite matrix.
    ///
    /// If the argument is `UPLO::Upper`, then computes the decomposition `A =
    /// U^H * U` using the upper triangular portion of `A` and returns the
    /// factorization containing `U`. Otherwise, if the argument is
    /// `UPLO::Lower`, computes the decomposition `A = L * L^H` using the lower
    /// triangular portion of `A` and returns the factorization containing `L`.
    fn cholesky_into(self, UPLO) -> Result<FactorizedCholesky<S>>;
}

/// Cholesky decomposition of Hermitian (or real symmetric) positive definite mutable reference of matrix
pub trait CholeskyMut<'a, S: Data> {
    /// Computes the Cholesky decomposition of the Hermitian (or real
    /// symmetric) positive definite matrix, storing the result (`L` or `U`
    /// according to the argument) in `self` and returning the factorization.
    ///
    /// If the argument is `UPLO::Upper`, then computes the decomposition `A =
    /// U^H * U` using the upper triangular portion of `A` and returns the
    /// factorization containing `U`. Otherwise, if the argument is
    /// `UPLO::Lower`, computes the decomposition `A = L * L^H` using the lower
    /// triangular portion of `A` and returns the factorization containing `L`.
    fn cholesky_mut(&'a mut self, UPLO) -> Result<FactorizedCholesky<S>>;
}

impl<A, S> CholeskyInto<S> for ArrayBase<S, Ix2>
where
    A: Scalar,
    S: DataMut<Elem = A>,
{
    fn cholesky_into(mut self, uplo: UPLO) -> Result<FactorizedCholesky<S>> {
        unsafe { A::cholesky(self.square_layout()?, uplo, self.as_allocated_mut()?)? };
        Ok(FactorizedCholesky {
            factor: self.into_triangular(uplo),
            uplo: uplo,
        })
    }
}

impl<'a, A, Si> CholeskyMut<'a, ViewRepr<&'a mut A>> for ArrayBase<Si, Ix2>
where
    A: Scalar,
    Si: DataMut<Elem = A>,
{
    fn cholesky_mut(&'a mut self, uplo: UPLO) -> Result<FactorizedCholesky<ViewRepr<&'a mut A>>> {
        unsafe { A::cholesky(self.square_layout()?, uplo, self.as_allocated_mut()?)? };
        Ok(FactorizedCholesky {
            factor: self.into_triangular(uplo).view_mut(),
            uplo: uplo,
        })
    }
}

impl<A, Si> Cholesky<OwnedRepr<A>> for ArrayBase<Si, Ix2>
where
    A: Scalar,
    Si: Data<Elem = A>,
{
    fn cholesky(&self, uplo: UPLO) -> Result<FactorizedCholesky<OwnedRepr<A>>> {
        let mut a = replicate(self);
        unsafe { A::cholesky(a.square_layout()?, uplo, a.as_allocated_mut()?)? };
        Ok(FactorizedCholesky {
            factor: a.into_triangular(uplo),
            uplo: uplo,
        })
    }
}
