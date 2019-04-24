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
//! use ndarray_linalg::cholesky::*;
//! # fn main() {
//!
//! let a: Array2<f64> = array![
//!     [  4.,  12., -16.],
//!     [ 12.,  37., -43.],
//!     [-16., -43.,  98.]
//! ];
//!
//! // Obtain `L`
//! let lower = a.cholesky(UPLO::Lower).unwrap();
//! assert!(lower.all_close(&array![
//!     [ 2., 0., 0.],
//!     [ 6., 1., 0.],
//!     [-8., 5., 3.]
//! ], 1e-9));
//!
//! // Find the determinant of `A`
//! let det = a.detc().unwrap();
//! assert!((det - 36.).abs() < 1e-9);
//!
//! // Solve `A * x = b`
//! let b = array![4., 13., -11.];
//! let x = a.solvec(&b).unwrap();
//! assert!(x.all_close(&array![-2., 1., 0.], 1e-9));
//! # }
//! ```

use ndarray::*;
use num_traits::Float;

use crate::convert::*;
use crate::error::*;
use crate::layout::*;
use crate::triangular::IntoTriangular;
use crate::types::*;

pub use crate::lapack_traits::UPLO;

/// Cholesky decomposition of Hermitian (or real symmetric) positive definite matrix
pub struct CholeskyFactorized<S: Data> {
    /// `L` from the decomposition `A = L * L^H` or `U` from the decomposition
    /// `A = U^H * U`.
    pub factor: ArrayBase<S, Ix2>,
    /// If this is `UPLO::Lower`, then `self.factor` is `L`. If this is
    /// `UPLO::Upper`, then `self.factor` is `U`.
    pub uplo: UPLO,
}

impl<A, S> CholeskyFactorized<S>
where
    A: Scalar + Lapack,
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
}

impl<A, S> DeterminantC for CholeskyFactorized<S>
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
{
    type Output = <A as Scalar>::Real;

    fn detc(&self) -> Self::Output {
        self.ln_detc().exp()
    }

    fn ln_detc(&self) -> Self::Output {
        self.factor
            .diag()
            .iter()
            .map(|elem| elem.square().ln())
            .sum::<Self::Output>()
    }
}

impl<A, S> DeterminantCInto for CholeskyFactorized<S>
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
{
    type Output = <A as Scalar>::Real;

    fn detc_into(self) -> Self::Output {
        self.detc()
    }

    fn ln_detc_into(self) -> Self::Output {
        self.ln_detc()
    }
}

impl<A, S> InverseC for CholeskyFactorized<S>
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
{
    type Output = Array2<A>;

    fn invc(&self) -> Result<Self::Output> {
        let f = CholeskyFactorized {
            factor: replicate(&self.factor),
            uplo: self.uplo,
        };
        f.invc_into()
    }
}

impl<A, S> InverseCInto for CholeskyFactorized<S>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
{
    type Output = ArrayBase<S, Ix2>;

    fn invc_into(self) -> Result<Self::Output> {
        let mut a = self.factor;
        unsafe { A::inv_cholesky(a.square_layout()?, self.uplo, a.as_allocated_mut()?)? };
        triangular_fill_hermitian(&mut a, self.uplo);
        Ok(a)
    }
}

impl<A, S> SolveC<A> for CholeskyFactorized<S>
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
{
    fn solvec_inplace<'a, Sb>(&self, b: &'a mut ArrayBase<Sb, Ix1>) -> Result<&'a mut ArrayBase<Sb, Ix1>>
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
    fn cholesky(&self, uplo: UPLO) -> Result<Self::Output>;
}

/// Cholesky decomposition of Hermitian (or real symmetric) positive definite matrix
pub trait CholeskyInto {
    type Output;
    /// Computes the Cholesky decomposition of the Hermitian (or real
    /// symmetric) positive definite matrix.
    ///
    /// If the argument is `UPLO::Upper`, then computes the decomposition `A =
    /// U^H * U` using the upper triangular portion of `A` and returns `U`.
    /// Otherwise, if the argument is `UPLO::Lower`, computes the decomposition
    /// `A = L * L^H` using the lower triangular portion of `A` and returns
    /// `L`.
    fn cholesky_into(self, uplo: UPLO) -> Result<Self::Output>;
}

/// Cholesky decomposition of Hermitian (or real symmetric) positive definite mutable reference of matrix
pub trait CholeskyInplace {
    /// Computes the Cholesky decomposition of the Hermitian (or real
    /// symmetric) positive definite matrix, writing the result (`L` or `U`
    /// according to the argument) to `self` and returning it.
    ///
    /// If the argument is `UPLO::Upper`, then computes the decomposition `A =
    /// U^H * U` using the upper triangular portion of `A` and writes `U`.
    /// Otherwise, if the argument is `UPLO::Lower`, computes the decomposition
    /// `A = L * L^H` using the lower triangular portion of `A` and writes `L`.
    fn cholesky_inplace(&mut self, uplo: UPLO) -> Result<&mut Self>;
}

impl<A, S> Cholesky for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
{
    type Output = Array2<A>;

    fn cholesky(&self, uplo: UPLO) -> Result<Array2<A>> {
        let a = replicate(self);
        a.cholesky_into(uplo)
    }
}

impl<A, S> CholeskyInto for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
{
    type Output = Self;

    fn cholesky_into(mut self, uplo: UPLO) -> Result<Self> {
        self.cholesky_inplace(uplo)?;
        Ok(self)
    }
}

impl<A, S> CholeskyInplace for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
{
    fn cholesky_inplace(&mut self, uplo: UPLO) -> Result<&mut Self> {
        unsafe { A::cholesky(self.square_layout()?, uplo, self.as_allocated_mut()?)? };
        Ok(self.into_triangular(uplo))
    }
}

/// Cholesky decomposition of Hermitian (or real symmetric) positive definite matrix reference
pub trait FactorizeC<S: Data> {
    /// Computes the Cholesky decomposition of the Hermitian (or real
    /// symmetric) positive definite matrix.
    ///
    /// If the argument is `UPLO::Upper`, then computes the decomposition `A =
    /// U^H * U` using the upper triangular portion of `A` and returns the
    /// factorization containing `U`. Otherwise, if the argument is
    /// `UPLO::Lower`, computes the decomposition `A = L * L^H` using the lower
    /// triangular portion of `A` and returns the factorization containing `L`.
    fn factorizec(&self, uplo: UPLO) -> Result<CholeskyFactorized<S>>;
}

/// Cholesky decomposition of Hermitian (or real symmetric) positive definite matrix
pub trait FactorizeCInto<S: Data> {
    /// Computes the Cholesky decomposition of the Hermitian (or real
    /// symmetric) positive definite matrix.
    ///
    /// If the argument is `UPLO::Upper`, then computes the decomposition `A =
    /// U^H * U` using the upper triangular portion of `A` and returns the
    /// factorization containing `U`. Otherwise, if the argument is
    /// `UPLO::Lower`, computes the decomposition `A = L * L^H` using the lower
    /// triangular portion of `A` and returns the factorization containing `L`.
    fn factorizec_into(self, uplo: UPLO) -> Result<CholeskyFactorized<S>>;
}

impl<A, S> FactorizeCInto<S> for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
{
    fn factorizec_into(self, uplo: UPLO) -> Result<CholeskyFactorized<S>> {
        Ok(CholeskyFactorized {
            factor: self.cholesky_into(uplo)?,
            uplo: uplo,
        })
    }
}

impl<A, Si> FactorizeC<OwnedRepr<A>> for ArrayBase<Si, Ix2>
where
    A: Scalar + Lapack,
    Si: Data<Elem = A>,
{
    fn factorizec(&self, uplo: UPLO) -> Result<CholeskyFactorized<OwnedRepr<A>>> {
        Ok(CholeskyFactorized {
            factor: self.cholesky(uplo)?,
            uplo: uplo,
        })
    }
}

/// Solve systems of linear equations with Hermitian (or real symmetric)
/// positive definite coefficient matrices
pub trait SolveC<A: Scalar> {
    /// Solves a system of linear equations `A * x = b` with Hermitian (or real
    /// symmetric) positive definite matrix `A`, where `A` is `self`, `b` is
    /// the argument, and `x` is the successful result.
    fn solvec<S: Data<Elem = A>>(&self, b: &ArrayBase<S, Ix1>) -> Result<Array1<A>> {
        let mut b = replicate(b);
        self.solvec_inplace(&mut b)?;
        Ok(b)
    }
    /// Solves a system of linear equations `A * x = b` with Hermitian (or real
    /// symmetric) positive definite matrix `A`, where `A` is `self`, `b` is
    /// the argument, and `x` is the successful result.
    fn solvec_into<S: DataMut<Elem = A>>(&self, mut b: ArrayBase<S, Ix1>) -> Result<ArrayBase<S, Ix1>> {
        self.solvec_inplace(&mut b)?;
        Ok(b)
    }
    /// Solves a system of linear equations `A * x = b` with Hermitian (or real
    /// symmetric) positive definite matrix `A`, where `A` is `self`, `b` is
    /// the argument, and `x` is the successful result. The value of `x` is
    /// also assigned to the argument.
    fn solvec_inplace<'a, S: DataMut<Elem = A>>(
        &self,
        b: &'a mut ArrayBase<S, Ix1>,
    ) -> Result<&'a mut ArrayBase<S, Ix1>>;
}

impl<A, S> SolveC<A> for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
{
    fn solvec_inplace<'a, Sb>(&self, b: &'a mut ArrayBase<Sb, Ix1>) -> Result<&'a mut ArrayBase<Sb, Ix1>>
    where
        Sb: DataMut<Elem = A>,
    {
        self.factorizec(UPLO::Upper)?.solvec_inplace(b)
    }
}

/// Inverse of Hermitian (or real symmetric) positive definite matrix ref
pub trait InverseC {
    type Output;
    /// Computes the inverse of the Hermitian (or real symmetric) positive
    /// definite matrix.
    fn invc(&self) -> Result<Self::Output>;
}

/// Inverse of Hermitian (or real symmetric) positive definite matrix
pub trait InverseCInto {
    type Output;
    /// Computes the inverse of the Hermitian (or real symmetric) positive
    /// definite matrix.
    fn invc_into(self) -> Result<Self::Output>;
}

impl<A, S> InverseC for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
{
    type Output = Array2<A>;

    fn invc(&self) -> Result<Self::Output> {
        self.factorizec(UPLO::Upper)?.invc_into()
    }
}

impl<A, S> InverseCInto for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
{
    type Output = Self;

    fn invc_into(self) -> Result<Self::Output> {
        self.factorizec_into(UPLO::Upper)?.invc_into()
    }
}

/// Determinant of Hermitian (or real symmetric) positive definite matrix ref
pub trait DeterminantC {
    type Output;

    /// Computes the determinant of the Hermitian (or real symmetric) positive
    /// definite matrix.
    fn detc(&self) -> Self::Output;

    /// Computes the natural log of the determinant of the Hermitian (or real
    /// symmetric) positive definite matrix.
    ///
    /// This method is more robust than `.detc()` to very small or very large
    /// determinants since it returns the natural logarithm of the determinant
    /// rather than the determinant itself.
    fn ln_detc(&self) -> Self::Output;
}

/// Determinant of Hermitian (or real symmetric) positive definite matrix
pub trait DeterminantCInto {
    type Output;

    /// Computes the determinant of the Hermitian (or real symmetric) positive
    /// definite matrix.
    fn detc_into(self) -> Self::Output;

    /// Computes the natural log of the determinant of the Hermitian (or real
    /// symmetric) positive definite matrix.
    ///
    /// This method is more robust than `.detc_into()` to very small or very
    /// large determinants since it returns the natural logarithm of the
    /// determinant rather than the determinant itself.
    fn ln_detc_into(self) -> Self::Output;
}

impl<A, S> DeterminantC for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
{
    type Output = Result<<A as Scalar>::Real>;

    fn detc(&self) -> Self::Output {
        Ok(self.ln_detc()?.exp())
    }

    fn ln_detc(&self) -> Self::Output {
        Ok(self.factorizec(UPLO::Upper)?.ln_detc())
    }
}

impl<A, S> DeterminantCInto for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
{
    type Output = Result<<A as Scalar>::Real>;

    fn detc_into(self) -> Self::Output {
        Ok(self.ln_detc_into()?.exp())
    }

    fn ln_detc_into(self) -> Self::Output {
        Ok(self.factorizec_into(UPLO::Upper)?.ln_detc_into())
    }
}
