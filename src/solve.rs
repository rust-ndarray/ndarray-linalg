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

use super::convert::*;
use super::error::*;
use super::layout::*;
use super::types::*;

pub use lapack_traits::{Pivot, Transpose};

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
/// * `*_mut` methods take a mutable reference to `b` and store the result in that array.
///
/// If you plan to solve many equations with the same `A` matrix but different
/// `b` vectors, it's faster to factor the `A` matrix once using the
/// `Factorize` trait, and then solve using the `Factorized` struct.
pub trait Solve<A: Scalar> {
    /// Solves a system of linear equations `A * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    fn solve<S: Data<Elem = A>>(&self, b: &ArrayBase<S, Ix1>) -> Result<Array1<A>> {
        let mut b = replicate(b);
        self.solve_mut(&mut b)?;
        Ok(b)
    }
    /// Solves a system of linear equations `A * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    fn solve_into<S: DataMut<Elem = A>>(&self, mut b: ArrayBase<S, Ix1>) -> Result<ArrayBase<S, Ix1>> {
        self.solve_mut(&mut b)?;
        Ok(b)
    }
    /// Solves a system of linear equations `A * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    fn solve_mut<'a, S: DataMut<Elem = A>>(&self, &'a mut ArrayBase<S, Ix1>) -> Result<&'a mut ArrayBase<S, Ix1>>;

    /// Solves a system of linear equations `A^T * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    fn solve_t<S: Data<Elem = A>>(&self, b: &ArrayBase<S, Ix1>) -> Result<Array1<A>> {
        let mut b = replicate(b);
        self.solve_t_mut(&mut b)?;
        Ok(b)
    }
    /// Solves a system of linear equations `A^T * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    fn solve_t_into<S: DataMut<Elem = A>>(&self, mut b: ArrayBase<S, Ix1>) -> Result<ArrayBase<S, Ix1>> {
        self.solve_t_mut(&mut b)?;
        Ok(b)
    }
    /// Solves a system of linear equations `A^T * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    fn solve_t_mut<'a, S: DataMut<Elem = A>>(&self, &'a mut ArrayBase<S, Ix1>) -> Result<&'a mut ArrayBase<S, Ix1>>;

    /// Solves a system of linear equations `A^H * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    fn solve_h<S: Data<Elem = A>>(&self, b: &ArrayBase<S, Ix1>) -> Result<Array1<A>> {
        let mut b = replicate(b);
        self.solve_h_mut(&mut b)?;
        Ok(b)
    }
    /// Solves a system of linear equations `A^H * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    fn solve_h_into<S: DataMut<Elem = A>>(&self, mut b: ArrayBase<S, Ix1>) -> Result<ArrayBase<S, Ix1>> {
        self.solve_h_mut(&mut b)?;
        Ok(b)
    }
    /// Solves a system of linear equations `A^H * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    fn solve_h_mut<'a, S: DataMut<Elem = A>>(&self, &'a mut ArrayBase<S, Ix1>) -> Result<&'a mut ArrayBase<S, Ix1>>;
}

/// Represents the LU factorization of a matrix `A` as `A = P*L*U`.
pub struct Factorized<S: Data> {
    /// The factors `L` and `U`; the unit diagonal elements of `L` are not
    /// stored.
    pub a: ArrayBase<S, Ix2>,
    /// The pivot indices that define the permutation matrix `P`.
    pub ipiv: Pivot,
}

impl<A, S> Solve<A> for Factorized<S>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    fn solve_mut<'a, Sb>(&self, mut rhs: &'a mut ArrayBase<Sb, Ix1>) -> Result<&'a mut ArrayBase<Sb, Ix1>>
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
    fn solve_t_mut<'a, Sb>(&self, mut rhs: &'a mut ArrayBase<Sb, Ix1>) -> Result<&'a mut ArrayBase<Sb, Ix1>>
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
    fn solve_h_mut<'a, Sb>(&self, mut rhs: &'a mut ArrayBase<Sb, Ix1>) -> Result<&'a mut ArrayBase<Sb, Ix1>>
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
    A: Scalar,
    S: Data<Elem = A>,
{
    fn solve_mut<'a, Sb>(&self, mut rhs: &'a mut ArrayBase<Sb, Ix1>) -> Result<&'a mut ArrayBase<Sb, Ix1>>
    where
        Sb: DataMut<Elem = A>,
    {
        let f = self.factorize()?;
        f.solve_mut(rhs)
    }
    fn solve_t_mut<'a, Sb>(&self, mut rhs: &'a mut ArrayBase<Sb, Ix1>) -> Result<&'a mut ArrayBase<Sb, Ix1>>
    where
        Sb: DataMut<Elem = A>,
    {
        let f = self.factorize()?;
        f.solve_t_mut(rhs)
    }
    fn solve_h_mut<'a, Sb>(&self, mut rhs: &'a mut ArrayBase<Sb, Ix1>) -> Result<&'a mut ArrayBase<Sb, Ix1>>
    where
        Sb: DataMut<Elem = A>,
    {
        let f = self.factorize()?;
        f.solve_h_mut(rhs)
    }
}


/// An interface for computing LU factorizations of matrix refs.
pub trait Factorize<S: Data> {
    /// Computes the LU factorization `A = P*L*U`, where `P` is a permutation
    /// matrix.
    fn factorize(&self) -> Result<Factorized<S>>;
}

/// An interface for computing LU factorizations of matrices.
pub trait FactorizeInto<S: Data> {
    /// Computes the LU factorization `A = P*L*U`, where `P` is a permutation
    /// matrix.
    fn factorize_into(self) -> Result<Factorized<S>>;
}

impl<A, S> FactorizeInto<S> for ArrayBase<S, Ix2>
where
    A: Scalar,
    S: DataMut<Elem = A>,
{
    fn factorize_into(mut self) -> Result<Factorized<S>> {
        let ipiv = unsafe { A::lu(self.layout()?, self.as_allocated_mut()?)? };
        Ok(Factorized {
            a: self,
            ipiv: ipiv,
        })
    }
}

impl<A, Si> Factorize<OwnedRepr<A>> for ArrayBase<Si, Ix2>
where
    A: Scalar,
    Si: Data<Elem = A>,
{
    fn factorize(&self) -> Result<Factorized<OwnedRepr<A>>> {
        let mut a: Array2<A> = replicate(self);
        let ipiv = unsafe { A::lu(a.layout()?, a.as_allocated_mut()?)? };
        Ok(Factorized { a: a, ipiv: ipiv })
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

impl<A, S> InverseInto for Factorized<S>
where
    A: Scalar,
    S: DataMut<Elem = A>,
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

impl<A, S> Inverse for Factorized<S>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    type Output = Array2<A>;

    fn inv(&self) -> Result<Array2<A>> {
        let f = Factorized {
            a: replicate(&self.a),
            ipiv: self.ipiv.clone(),
        };
        f.inv_into()
    }
}

impl<A, S> InverseInto for ArrayBase<S, Ix2>
where
    A: Scalar,
    S: DataMut<Elem = A>,
{
    type Output = Self;

    fn inv_into(self) -> Result<Self::Output> {
        let f = self.factorize_into()?;
        f.inv_into()
    }
}

impl<A, Si> Inverse for ArrayBase<Si, Ix2>
where
    A: Scalar,
    Si: Data<Elem = A>,
{
    type Output = Array2<A>;

    fn inv(&self) -> Result<Self::Output> {
        let f = self.factorize()?;
        f.inv_into()
    }
}
