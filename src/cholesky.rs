//! Cholesky decomposition of Hermitian (or real symmetric) positive definite matrices
//!
//! See the [Wikipedia page about Cholesky
//! decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition) for
//! more information.
//!
//! # Example
//!
//! Calculate `L` in the Cholesky decomposition `A = L * L^H`, where `A` is a
//! Hermitian (or real symmetric) positive definite matrix:
//!
//! ```
//! #[macro_use]
//! extern crate ndarray;
//! extern crate ndarray_linalg;
//!
//! use ndarray::prelude::*;
//! use ndarray_linalg::{CholeskyInto, UPLO};
//! # fn main() {
//!
//! let a: Array2<f64> = array![
//!     [  4.,  12., -16.],
//!     [ 12.,  37., -43.],
//!     [-16., -43.,  98.]
//! ];
//! let lower = a.cholesky_into(UPLO::Lower).unwrap();
//! assert!(lower.all_close(&array![
//!     [ 2., 0., 0.],
//!     [ 6., 1., 0.],
//!     [-8., 5., 3.]
//! ], 1e-9));
//! # }
//! ```

use ndarray::*;

use super::convert::*;
use super::error::*;
use super::layout::*;
use super::triangular::IntoTriangular;
use super::types::*;

pub use lapack_traits::UPLO;

/// Cholesky decomposition of Hermitian (or real symmetric) positive definite matrix reference
pub trait Cholesky {
    type Output;
    /// Computes the Cholesky decomposition of the Hermitian (or real
    /// symmetric) positive definite matrix.
    ///
    /// If the argument is `UPLO::Upper`, then computes the decomposition `A =
    /// U^H * U` using the upper triangular portion of `A` and returns `U`.
    /// Otherwise, if the argument is `UPLO::Lower`, computes the decomposition
    /// `A = L * L^H` using the lower triangular portion of `A` and returns
    /// `L`.
    fn cholesky(&self, UPLO) -> Result<Self::Output>;
}

/// Cholesky decomposition of Hermitian (or real symmetric) positive definite matrix
pub trait CholeskyInto: Sized {
    /// Computes the Cholesky decomposition of the Hermitian (or real
    /// symmetric) positive definite matrix.
    ///
    /// If the argument is `UPLO::Upper`, then computes the decomposition `A =
    /// U^H * U` using the upper triangular portion of `A` and returns `U`.
    /// Otherwise, if the argument is `UPLO::Lower`, computes the decomposition
    /// `A = L * L^H` using the lower triangular portion of `A` and returns
    /// `L`.
    fn cholesky_into(self, UPLO) -> Result<Self>;
}

/// Cholesky decomposition of Hermitian (or real symmetric) positive definite mutable reference of matrix
pub trait CholeskyMut {
    /// Computes the Cholesky decomposition of the Hermitian (or real
    /// symmetric) positive definite matrix, storing the result in `self` and
    /// returning it.
    ///
    /// If the argument is `UPLO::Upper`, then computes the decomposition `A =
    /// U^H * U` using the upper triangular portion of `A` and returns `U`.
    /// Otherwise, if the argument is `UPLO::Lower`, computes the decomposition
    /// `A = L * L^H` using the lower triangular portion of `A` and returns
    /// `L`.
    fn cholesky_mut(&mut self, UPLO) -> Result<&mut Self>;
}

impl<A, S> CholeskyInto for ArrayBase<S, Ix2>
where
    A: Scalar,
    S: DataMut<Elem = A>,
{
    fn cholesky_into(mut self, uplo: UPLO) -> Result<Self> {
        unsafe { A::cholesky(self.square_layout()?, uplo, self.as_allocated_mut()?)? };
        Ok(self.into_triangular(uplo))
    }
}

impl<A, S> CholeskyMut for ArrayBase<S, Ix2>
where
    A: Scalar,
    S: DataMut<Elem = A>,
{
    fn cholesky_mut(&mut self, uplo: UPLO) -> Result<&mut Self> {
        unsafe { A::cholesky(self.square_layout()?, uplo, self.as_allocated_mut()?)? };
        Ok(self.into_triangular(uplo))
    }
}

impl<A, S> Cholesky for ArrayBase<S, Ix2>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    type Output = Array2<A>;

    fn cholesky(&self, uplo: UPLO) -> Result<Self::Output> {
        let mut a = replicate(self);
        unsafe { A::cholesky(a.square_layout()?, uplo, a.as_allocated_mut()?)? };
        Ok(a.into_triangular(uplo))
    }
}
