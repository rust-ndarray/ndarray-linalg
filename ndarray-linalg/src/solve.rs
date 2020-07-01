//! Solve systems of linear equations and invert matrices
//!
//! # Examples
//!
//! Solve `A * x = b`:
//!
//! ```
//! #[macro_use]
//! extern crate ndarray;
//! extern crate ndarray_linalg;
//!
//! use ndarray::prelude::*;
//! use ndarray_linalg::Solve;
//! # fn main() {
//!
//! let a: Array2<f64> = array![[3., 2., -1.], [2., -2., 4.], [-2., 1., -2.]];
//! let b: Array1<f64> = array![1., -2., 0.];
//! let x = a.solve_into(b).unwrap();
//! assert!(x.all_close(&array![1., -2., -2.], 1e-9));
//!
//! # }
//! ```
//!
//! There are also special functions for solving `A^T * x = b` and
//! `A^H * x = b`.
//!
//! If you are solving multiple systems of linear equations with the same
//! coefficient matrix `A`, it's faster to compute the LU factorization once at
//! the beginning than solving directly using `A`:
//!
//! ```
//! # extern crate ndarray;
//! # extern crate ndarray_linalg;
//!
//! use ndarray::prelude::*;
//! use ndarray_linalg::*;
//! # fn main() {
//!
//! let a: Array2<f64> = random((3, 3));
//! let f = a.factorize_into().unwrap(); // LU factorize A (A is consumed)
//! for _ in 0..10 {
//!     let b: Array1<f64> = random(3);
//!     let x = f.solve_into(b).unwrap(); // Solve A * x = b using factorized L, U
//! }
//!
//! # }
//! ```

use ndarray::*;
use num_traits::{Float, Zero};

use crate::convert::*;
use crate::error::*;
use crate::layout::*;
use crate::opnorm::OperationNorm;
use crate::types::*;

pub use crate::lapack::{Pivot, Transpose};

/// An interface for solving systems of linear equations.
///
/// There are three groups of methods:
///
/// * `solve*` (normal) methods solve `A * x = b` for `x`.
/// * `solve_t*` (transpose) methods solve `A^T * x = b` for `x`.
/// * `solve_h*` (Hermitian conjugate) methods solve `A^H * x = b` for `x`.
///
/// Within each group, there are three methods that handle ownership differently:
///
/// * `*` methods take a reference to `b` and return `x` as a new array.
/// * `*_into` methods take ownership of `b`, store the result in it, and return it.
/// * `*_inplace` methods take a mutable reference to `b` and store the result in that array.
///
/// If you plan to solve many equations with the same `A` matrix but different
/// `b` vectors, it's faster to factor the `A` matrix once using the
/// `Factorize` trait, and then solve using the `LUFactorized` struct.
pub trait Solve<A: Scalar> {
    /// Solves a system of linear equations `A * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    fn solve<S: Data<Elem = A>>(&self, b: &ArrayBase<S, Ix1>) -> Result<Array1<A>> {
        let mut b = replicate(b);
        self.solve_inplace(&mut b)?;
        Ok(b)
    }
    /// Solves a system of linear equations `A * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    fn solve_into<S: DataMut<Elem = A>>(
        &self,
        mut b: ArrayBase<S, Ix1>,
    ) -> Result<ArrayBase<S, Ix1>> {
        self.solve_inplace(&mut b)?;
        Ok(b)
    }
    /// Solves a system of linear equations `A * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    fn solve_inplace<'a, S: DataMut<Elem = A>>(
        &self,
        b: &'a mut ArrayBase<S, Ix1>,
    ) -> Result<&'a mut ArrayBase<S, Ix1>>;

    /// Solves a system of linear equations `A^T * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    fn solve_t<S: Data<Elem = A>>(&self, b: &ArrayBase<S, Ix1>) -> Result<Array1<A>> {
        let mut b = replicate(b);
        self.solve_t_inplace(&mut b)?;
        Ok(b)
    }
    /// Solves a system of linear equations `A^T * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    fn solve_t_into<S: DataMut<Elem = A>>(
        &self,
        mut b: ArrayBase<S, Ix1>,
    ) -> Result<ArrayBase<S, Ix1>> {
        self.solve_t_inplace(&mut b)?;
        Ok(b)
    }
    /// Solves a system of linear equations `A^T * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    fn solve_t_inplace<'a, S: DataMut<Elem = A>>(
        &self,
        b: &'a mut ArrayBase<S, Ix1>,
    ) -> Result<&'a mut ArrayBase<S, Ix1>>;

    /// Solves a system of linear equations `A^H * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    fn solve_h<S: Data<Elem = A>>(&self, b: &ArrayBase<S, Ix1>) -> Result<Array1<A>> {
        let mut b = replicate(b);
        self.solve_h_inplace(&mut b)?;
        Ok(b)
    }
    /// Solves a system of linear equations `A^H * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    fn solve_h_into<S: DataMut<Elem = A>>(
        &self,
        mut b: ArrayBase<S, Ix1>,
    ) -> Result<ArrayBase<S, Ix1>> {
        self.solve_h_inplace(&mut b)?;
        Ok(b)
    }
    /// Solves a system of linear equations `A^H * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    fn solve_h_inplace<'a, S: DataMut<Elem = A>>(
        &self,
        b: &'a mut ArrayBase<S, Ix1>,
    ) -> Result<&'a mut ArrayBase<S, Ix1>>;
}

/// Represents the LU factorization of a matrix `A` as `A = P*L*U`.
#[derive(Clone)]
pub struct LUFactorized<S: Data + RawDataClone> {
    /// The factors `L` and `U`; the unit diagonal elements of `L` are not
    /// stored.
    pub a: ArrayBase<S, Ix2>,
    /// The pivot indices that define the permutation matrix `P`.
    pub ipiv: Pivot,
}

impl<A, S> Solve<A> for LUFactorized<S>
where
    A: Scalar + Lapack,
    S: Data<Elem = A> + RawDataClone,
{
    fn solve_inplace<'a, Sb>(
        &self,
        rhs: &'a mut ArrayBase<Sb, Ix1>,
    ) -> Result<&'a mut ArrayBase<Sb, Ix1>>
    where
        Sb: DataMut<Elem = A>,
    {
        unsafe {
            A::solve(
                self.a.square_layout()?,
                Transpose::No,
                self.a.as_allocated()?,
                &self.ipiv,
                rhs.as_slice_mut().unwrap(),
            )?
        };
        Ok(rhs)
    }
    fn solve_t_inplace<'a, Sb>(
        &self,
        rhs: &'a mut ArrayBase<Sb, Ix1>,
    ) -> Result<&'a mut ArrayBase<Sb, Ix1>>
    where
        Sb: DataMut<Elem = A>,
    {
        unsafe {
            A::solve(
                self.a.square_layout()?,
                Transpose::Transpose,
                self.a.as_allocated()?,
                &self.ipiv,
                rhs.as_slice_mut().unwrap(),
            )?
        };
        Ok(rhs)
    }
    fn solve_h_inplace<'a, Sb>(
        &self,
        rhs: &'a mut ArrayBase<Sb, Ix1>,
    ) -> Result<&'a mut ArrayBase<Sb, Ix1>>
    where
        Sb: DataMut<Elem = A>,
    {
        unsafe {
            A::solve(
                self.a.square_layout()?,
                Transpose::Hermite,
                self.a.as_allocated()?,
                &self.ipiv,
                rhs.as_slice_mut().unwrap(),
            )?
        };
        Ok(rhs)
    }
}

impl<A, S> Solve<A> for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
{
    fn solve_inplace<'a, Sb>(
        &self,
        rhs: &'a mut ArrayBase<Sb, Ix1>,
    ) -> Result<&'a mut ArrayBase<Sb, Ix1>>
    where
        Sb: DataMut<Elem = A>,
    {
        let f = self.factorize()?;
        f.solve_inplace(rhs)
    }
    fn solve_t_inplace<'a, Sb>(
        &self,
        rhs: &'a mut ArrayBase<Sb, Ix1>,
    ) -> Result<&'a mut ArrayBase<Sb, Ix1>>
    where
        Sb: DataMut<Elem = A>,
    {
        let f = self.factorize()?;
        f.solve_t_inplace(rhs)
    }
    fn solve_h_inplace<'a, Sb>(
        &self,
        rhs: &'a mut ArrayBase<Sb, Ix1>,
    ) -> Result<&'a mut ArrayBase<Sb, Ix1>>
    where
        Sb: DataMut<Elem = A>,
    {
        let f = self.factorize()?;
        f.solve_h_inplace(rhs)
    }
}

/// An interface for computing LU factorizations of matrix refs.
pub trait Factorize<S: Data + RawDataClone> {
    /// Computes the LU factorization `A = P*L*U`, where `P` is a permutation
    /// matrix.
    fn factorize(&self) -> Result<LUFactorized<S>>;
}

/// An interface for computing LU factorizations of matrices.
pub trait FactorizeInto<S: Data + RawDataClone> {
    /// Computes the LU factorization `A = P*L*U`, where `P` is a permutation
    /// matrix.
    fn factorize_into(self) -> Result<LUFactorized<S>>;
}

impl<A, S> FactorizeInto<S> for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A> + RawDataClone,
{
    fn factorize_into(mut self) -> Result<LUFactorized<S>> {
        let ipiv = unsafe { A::lu(self.layout()?, self.as_allocated_mut()?)? };
        Ok(LUFactorized { a: self, ipiv })
    }
}

impl<A, Si> Factorize<OwnedRepr<A>> for ArrayBase<Si, Ix2>
where
    A: Scalar + Lapack,
    Si: Data<Elem = A>,
{
    fn factorize(&self) -> Result<LUFactorized<OwnedRepr<A>>> {
        let mut a: Array2<A> = replicate(self);
        let ipiv = unsafe { A::lu(a.layout()?, a.as_allocated_mut()?)? };
        Ok(LUFactorized { a, ipiv })
    }
}

/// An interface for inverting matrix refs.
pub trait Inverse {
    type Output;
    /// Computes the inverse of the matrix.
    fn inv(&self) -> Result<Self::Output>;
}

/// An interface for inverting matrices.
pub trait InverseInto {
    type Output;
    /// Computes the inverse of the matrix.
    fn inv_into(self) -> Result<Self::Output>;
}

impl<A, S> InverseInto for LUFactorized<S>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A> + RawDataClone,
{
    type Output = ArrayBase<S, Ix2>;

    fn inv_into(mut self) -> Result<ArrayBase<S, Ix2>> {
        unsafe {
            A::inv(
                self.a.square_layout()?,
                self.a.as_allocated_mut()?,
                &self.ipiv,
            )?
        };
        Ok(self.a)
    }
}

impl<A, S> Inverse for LUFactorized<S>
where
    A: Scalar + Lapack,
    S: Data<Elem = A> + RawDataClone,
{
    type Output = Array2<A>;

    fn inv(&self) -> Result<Array2<A>> {
        let f = LUFactorized {
            a: replicate(&self.a),
            ipiv: self.ipiv.clone(),
        };
        f.inv_into()
    }
}

impl<A, S> InverseInto for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A> + RawDataClone,
{
    type Output = Self;

    fn inv_into(self) -> Result<Self::Output> {
        let f = self.factorize_into()?;
        f.inv_into()
    }
}

impl<A, Si> Inverse for ArrayBase<Si, Ix2>
where
    A: Scalar + Lapack,
    Si: Data<Elem = A>,
{
    type Output = Array2<A>;

    fn inv(&self) -> Result<Self::Output> {
        let f = self.factorize()?;
        f.inv_into()
    }
}

/// An interface for calculating determinants of matrix refs.
pub trait Determinant<A: Scalar> {
    /// Computes the determinant of the matrix.
    fn det(&self) -> Result<A> {
        let (sign, ln_det) = self.sln_det()?;
        Ok(sign * A::from_real(ln_det.exp()))
    }

    /// Computes the `(sign, natural_log)` of the determinant of the matrix.
    ///
    /// For real matrices, `sign` is `1`, `0`, or `-1`. For complex matrices,
    /// `sign` is `0` or a complex number with absolute value 1. The
    /// `natural_log` is the natural logarithm of the absolute value of the
    /// determinant. If the determinant is zero, `sign` is 0 and `natural_log`
    /// is negative infinity.
    ///
    /// To obtain the determinant, you can compute `sign * natural_log.exp()`
    /// or just call `.det()` instead.
    ///
    /// This method is more robust than `.det()` to very small or very large
    /// determinants since it returns the natural logarithm of the determinant
    /// rather than the determinant itself.
    fn sln_det(&self) -> Result<(A, A::Real)>;
}

/// An interface for calculating determinants of matrices.
pub trait DeterminantInto<A: Scalar>: Sized {
    /// Computes the determinant of the matrix.
    fn det_into(self) -> Result<A> {
        let (sign, ln_det) = self.sln_det_into()?;
        Ok(sign * A::from_real(ln_det.exp()))
    }

    /// Computes the `(sign, natural_log)` of the determinant of the matrix.
    ///
    /// For real matrices, `sign` is `1`, `0`, or `-1`. For complex matrices,
    /// `sign` is `0` or a complex number with absolute value 1. The
    /// `natural_log` is the natural logarithm of the absolute value of the
    /// determinant. If the determinant is zero, `sign` is 0 and `natural_log`
    /// is negative infinity.
    ///
    /// To obtain the determinant, you can compute `sign * natural_log.exp()`
    /// or just call `.det_into()` instead.
    ///
    /// This method is more robust than `.det()` to very small or very large
    /// determinants since it returns the natural logarithm of the determinant
    /// rather than the determinant itself.
    fn sln_det_into(self) -> Result<(A, A::Real)>;
}

fn lu_sln_det<'a, A, P, U>(ipiv_iter: P, u_diag_iter: U) -> (A, A::Real)
where
    A: Scalar + Lapack,
    P: Iterator<Item = i32>,
    U: Iterator<Item = &'a A>,
{
    let pivot_sign = if ipiv_iter
        .enumerate()
        .filter(|&(i, pivot)| pivot != i as i32 + 1)
        .count()
        % 2
        == 0
    {
        A::one()
    } else {
        -A::one()
    };
    let (upper_sign, ln_det) = u_diag_iter.fold(
        (A::one(), A::Real::zero()),
        |(upper_sign, ln_det), &elem| {
            let abs_elem: A::Real = elem.abs();
            (
                upper_sign * elem / A::from_real(abs_elem),
                ln_det + abs_elem.ln(),
            )
        },
    );
    (pivot_sign * upper_sign, ln_det)
}

impl<A, S> Determinant<A> for LUFactorized<S>
where
    A: Scalar + Lapack,
    S: Data<Elem = A> + RawDataClone,
{
    fn sln_det(&self) -> Result<(A, A::Real)> {
        self.a.ensure_square()?;
        Ok(lu_sln_det(self.ipiv.iter().cloned(), self.a.diag().iter()))
    }
}

impl<A, S> DeterminantInto<A> for LUFactorized<S>
where
    A: Scalar + Lapack,
    S: Data<Elem = A> + RawDataClone,
{
    fn sln_det_into(self) -> Result<(A, A::Real)> {
        self.a.ensure_square()?;
        Ok(lu_sln_det(self.ipiv.into_iter(), self.a.into_diag().iter()))
    }
}

impl<A, S> Determinant<A> for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
{
    fn sln_det(&self) -> Result<(A, A::Real)> {
        self.ensure_square()?;
        match self.factorize() {
            Ok(fac) => fac.sln_det(),
            Err(LinalgError::LapackComputationalFailure { .. }) => {
                // The determinant is zero.
                Ok((A::zero(), A::Real::neg_infinity()))
            }
            Err(err) => Err(err),
        }
    }
}

impl<A, S> DeterminantInto<A> for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A> + RawDataClone,
{
    fn sln_det_into(self) -> Result<(A, A::Real)> {
        self.ensure_square()?;
        match self.factorize_into() {
            Ok(fac) => fac.sln_det_into(),
            Err(LinalgError::LapackComputationalFailure { .. }) => {
                // The determinant is zero.
                Ok((A::zero(), A::Real::neg_infinity()))
            }
            Err(err) => Err(err),
        }
    }
}

/// An interface for *estimating* the reciprocal condition number of matrix refs.
pub trait ReciprocalConditionNum<A: Scalar> {
    /// *Estimates* the reciprocal of the condition number of the matrix in
    /// 1-norm.
    ///
    /// This method uses the LAPACK `*gecon` routines, which *estimate*
    /// `self.inv().opnorm_one()` and then compute `rcond = 1. /
    /// (self.opnorm_one() * self.inv().opnorm_one())`.
    ///
    /// * If `rcond` is near `0.`, the matrix is badly conditioned.
    /// * If `rcond` is near `1.`, the matrix is well conditioned.
    fn rcond(&self) -> Result<A::Real>;
}

/// An interface for *estimating* the reciprocal condition number of matrices.
pub trait ReciprocalConditionNumInto<A: Scalar> {
    /// *Estimates* the reciprocal of the condition number of the matrix in
    /// 1-norm.
    ///
    /// This method uses the LAPACK `*gecon` routines, which *estimate*
    /// `self.inv().opnorm_one()` and then compute `rcond = 1. /
    /// (self.opnorm_one() * self.inv().opnorm_one())`.
    ///
    /// * If `rcond` is near `0.`, the matrix is badly conditioned.
    /// * If `rcond` is near `1.`, the matrix is well conditioned.
    fn rcond_into(self) -> Result<A::Real>;
}

impl<A, S> ReciprocalConditionNum<A> for LUFactorized<S>
where
    A: Scalar + Lapack,
    S: Data<Elem = A> + RawDataClone,
{
    fn rcond(&self) -> Result<A::Real> {
        unsafe {
            A::rcond(
                self.a.layout()?,
                self.a.as_allocated()?,
                self.a.opnorm_one()?,
            )
        }
    }
}

impl<A, S> ReciprocalConditionNumInto<A> for LUFactorized<S>
where
    A: Scalar + Lapack,
    S: Data<Elem = A> + RawDataClone,
{
    fn rcond_into(self) -> Result<A::Real> {
        self.rcond()
    }
}

impl<A, S> ReciprocalConditionNum<A> for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
{
    fn rcond(&self) -> Result<A::Real> {
        self.factorize()?.rcond_into()
    }
}

impl<A, S> ReciprocalConditionNumInto<A> for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A> + RawDataClone,
{
    fn rcond_into(self) -> Result<A::Real> {
        self.factorize_into()?.rcond_into()
    }
}
