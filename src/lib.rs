//! This crate implements matrix manipulation for
//! [rust-ndarray](https://github.com/bluss/rust-ndarray) using LAPACK.
//!
//! Basic manipulations are implemented as matrix traits,
//! [Matrix](matrix/trait.Matrix.html), [SquareMatrix](square/trait.SquareMatrix.html),
//! and [HermiteMatrix](hermite/trait.HermiteMatrix.html).
//!
//! Matrix
//! -------
//! - [singular-value decomposition](matrix/trait.Matrix.html#tymethod.svd)
//! - [LU decomposition](matrix/trait.Matrix.html#tymethod.lu)
//! - [QR decomposition](matrix/trait.Matrix.html#tymethod.qr)
//! - [operator norm for L1 norm](matrix/trait.Matrix.html#tymethod.norm_1)
//! - [operator norm for L-inf norm](matrix/trait.Matrix.html#tymethod.norm_i)
//! - [Frobeiuns norm](matrix/trait.Matrix.html#tymethod.norm_f)
//!
//! SquareMatrix
//! -------------
//! - [inverse of matrix](square/trait.SquareMatrix.html#tymethod.inv)
//! - [trace of matrix](square/trait.SquareMatrix.html#tymethod.trace)
//! - [WIP] eigenvalue
//!
//! HermiteMatrix
//! --------------
//! - [eigenvalue analysis](hermite/trait.HermiteMatrix.html#tymethod.eigh)
//! - [symmetric square root](hermite/trait.HermiteMatrix.html#tymethod.ssqrt)
//! - [Cholesky factorization](hermite/trait.HermiteMatrix.html#tymethod.cholesky)
//!
//! Others
//! -------
//! - [solve triangular](triangular/trait.SolveTriangular.html)
//! - [misc utilities](util/index.html)

extern crate blas;
extern crate lapack;
extern crate num_traits;
extern crate num_complex;
#[macro_use(s)]
extern crate ndarray;
#[macro_use]
extern crate enum_error_derive;
#[macro_use]
extern crate derive_new;

pub mod impls;
pub mod error;

pub mod vector;
pub mod matrix;
pub mod square;
pub mod hermite;
pub mod triangular;

pub mod util;
pub mod assert;
pub mod prelude;
