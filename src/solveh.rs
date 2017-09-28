//! Solve Hermitian (or real symmetric) linear problems and invert Hermitian
//! (or real symmetric) matrices
//!
//! **Note that only the upper triangular portion of the matrix is used.**
//!
//! # Examples
//!
//! Solve `A * x = b`, where `A` is a Hermitian (or real symmetric) matrix:
//!
//! ```
//! #[macro_use]
//! extern crate ndarray;
//! extern crate ndarray_linalg;
//!
//! use ndarray::prelude::*;
//! use ndarray_linalg::SolveH;
//! # fn main() {
//!
//! let a: Array2<f64> = array![
//!     [3., 2., -1.],
//!     [2., -2., 4.],
//!     [-1., 4., 5.]
//! ];
//! let b: Array1<f64> = array![11., -12., 1.];
//! let x = a.solveh_into(b).unwrap();
//! assert!(x.all_close(&array![1., 3., -2.], 1e-9));
//!
//! # }
//! ```
//!
//! If you are solving multiple systems of linear equations with the same
//! Hermitian or real symmetric coefficient matrix `A`, it's faster to compute
//! the factorization once at the beginning than solving directly using `A`:
//!
//! ```
//! # extern crate ndarray;
//! # extern crate ndarray_linalg;
//! use ndarray::prelude::*;
//! use ndarray_linalg::*;
//! # fn main() {
//!
//! let a: Array2<f64> = random((3, 3));
//! let f = a.factorizeh_into().unwrap(); // Factorize A (A is consumed)
//! for _ in 0..10 {
//!     let b: Array1<f64> = random(3);
//!     let x = f.solveh_into(b).unwrap(); // Solve A * x = b using the factorization
//! }
//!
//! # }
//! ```

use ndarray::*;

use super::convert::*;
use super::error::*;
use super::layout::*;
use super::types::*;

pub use lapack_traits::{Pivot, UPLO};

/// An interface for solving systems of Hermitian (or real symmetric) linear equations.
///
/// If you plan to solve many equations with the same Hermitian (or real
/// symmetric) coefficient matrix `A` but different `b` vectors, it's faster to
/// factor the `A` matrix once using the `FactorizeH` trait, and then solve
/// using the `BKFactorized` struct.
pub trait SolveH<A: Scalar> {
    /// Solves a system of linear equations `A * x = b` with Hermitian (or real
    /// symmetric) matrix `A`, where `A` is `self`, `b` is the argument, and
    /// `x` is the successful result.
    fn solveh<S: Data<Elem = A>>(&self, b: &ArrayBase<S, Ix1>) -> Result<Array1<A>> {
        let mut b = replicate(b);
        self.solveh_inplace(&mut b)?;
        Ok(b)
    }
    /// Solves a system of linear equations `A * x = b` with Hermitian (or real
    /// symmetric) matrix `A`, where `A` is `self`, `b` is the argument, and
    /// `x` is the successful result.
    fn solveh_into<S: DataMut<Elem = A>>(&self, mut b: ArrayBase<S, Ix1>) -> Result<ArrayBase<S, Ix1>> {
        self.solveh_inplace(&mut b)?;
        Ok(b)
    }
    /// Solves a system of linear equations `A * x = b` with Hermitian (or real
    /// symmetric) matrix `A`, where `A` is `self`, `b` is the argument, and
    /// `x` is the successful result. The value of `x` is also assigned to the
    /// argument.
    fn solveh_inplace<'a, S: DataMut<Elem = A>>(&self, &'a mut ArrayBase<S, Ix1>) -> Result<&'a mut ArrayBase<S, Ix1>>;
}

/// Represents the Bunch–Kaufman factorization of a Hermitian (or real
/// symmetric) matrix as `A = P * U * D * U^H * P^T`.
pub struct BKFactorized<S: Data> {
    pub a: ArrayBase<S, Ix2>,
    pub ipiv: Pivot,
}

impl<A, S> SolveH<A> for BKFactorized<S>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    fn solveh_inplace<'a, Sb>(&self, rhs: &'a mut ArrayBase<Sb, Ix1>) -> Result<&'a mut ArrayBase<Sb, Ix1>>
    where
        Sb: DataMut<Elem = A>,
    {
        unsafe {
            A::solveh(
                self.a.square_layout()?,
                UPLO::Upper,
                self.a.as_allocated()?,
                &self.ipiv,
                rhs.as_slice_mut().unwrap(),
            )?
        };
        Ok(rhs)
    }
}

impl<A, S> SolveH<A> for ArrayBase<S, Ix2>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    fn solveh_inplace<'a, Sb>(&self, mut rhs: &'a mut ArrayBase<Sb, Ix1>) -> Result<&'a mut ArrayBase<Sb, Ix1>>
    where
        Sb: DataMut<Elem = A>,
    {
        let f = self.factorizeh()?;
        f.solveh_inplace(rhs)
    }
}


/// An interface for computing the Bunch–Kaufman factorization of Hermitian (or
/// real symmetric) matrix refs.
pub trait FactorizeH<S: Data> {
    /// Computes the Bunch–Kaufman factorization of a Hermitian (or real
    /// symmetric) matrix.
    fn factorizeh(&self) -> Result<BKFactorized<S>>;
}

/// An interface for computing the Bunch–Kaufman factorization of Hermitian (or
/// real symmetric) matrices.
pub trait FactorizeHInto<S: Data> {
    /// Computes the Bunch–Kaufman factorization of a Hermitian (or real
    /// symmetric) matrix.
    fn factorizeh_into(self) -> Result<BKFactorized<S>>;
}

impl<A, S> FactorizeHInto<S> for ArrayBase<S, Ix2>
where
    A: Scalar,
    S: DataMut<Elem = A>,
{
    fn factorizeh_into(mut self) -> Result<BKFactorized<S>> {
        let ipiv = unsafe { A::bk(self.layout()?, UPLO::Upper, self.as_allocated_mut()?)? };
        Ok(BKFactorized {
            a: self,
            ipiv: ipiv,
        })
    }
}

impl<A, Si> FactorizeH<OwnedRepr<A>> for ArrayBase<Si, Ix2>
where
    A: Scalar,
    Si: Data<Elem = A>,
{
    fn factorizeh(&self) -> Result<BKFactorized<OwnedRepr<A>>> {
        let mut a: Array2<A> = replicate(self);
        let ipiv = unsafe { A::bk(a.layout()?, UPLO::Upper, a.as_allocated_mut()?)? };
        Ok(BKFactorized { a: a, ipiv: ipiv })
    }
}

/// An interface for inverting Hermitian (or real symmetric) matrix refs.
pub trait InverseH {
    type Output;
    /// Computes the inverse of the Hermitian (or real symmetric) matrix.
    ///
    /// **Warning: The inverse is stored only in the upper triangular portion
    /// of the result matrix!** If you want the lower triangular portion to be
    /// correct, you must fill it in according to the results in the upper
    /// triangular portion.
    fn invh(&self) -> Result<Self::Output>;
}

/// An interface for inverting Hermitian (or real symmetric) matrices.
pub trait InverseHInto {
    type Output;
    /// Computes the inverse of the Hermitian (or real symmetric) matrix.
    ///
    /// **Warning: The inverse is stored only in the upper triangular portion
    /// of the result matrix!** If you want the lower triangular portion to be
    /// correct, you must fill it in according to the results in the upper
    /// triangular portion.
    fn invh_into(self) -> Result<Self::Output>;
}

impl<A, S> InverseHInto for BKFactorized<S>
where
    A: Scalar,
    S: DataMut<Elem = A>,
{
    type Output = ArrayBase<S, Ix2>;

    fn invh_into(mut self) -> Result<ArrayBase<S, Ix2>> {
        unsafe {
            A::invh(
                self.a.square_layout()?,
                UPLO::Upper,
                self.a.as_allocated_mut()?,
                &self.ipiv,
            )?
        };
        Ok(self.a)
    }
}

impl<A, S> InverseH for BKFactorized<S>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    type Output = Array2<A>;

    fn invh(&self) -> Result<Self::Output> {
        let f = BKFactorized {
            a: replicate(&self.a),
            ipiv: self.ipiv.clone(),
        };
        f.invh_into()
    }
}

impl<A, S> InverseHInto for ArrayBase<S, Ix2>
where
    A: Scalar,
    S: DataMut<Elem = A>,
{
    type Output = Self;

    fn invh_into(self) -> Result<Self::Output> {
        let f = self.factorizeh_into()?;
        f.invh_into()
    }
}

impl<A, Si> InverseH for ArrayBase<Si, Ix2>
where
    A: Scalar,
    Si: Data<Elem = A>,
{
    type Output = Array2<A>;

    fn invh(&self) -> Result<Self::Output> {
        let f = self.factorizeh()?;
        f.invh_into()
    }
}
