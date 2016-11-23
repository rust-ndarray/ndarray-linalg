//! This crate serves matrix manipulation for [rust-ndarray](https://github.com/bluss/rust-ndarray)
//! They are implemented as traits,
//! [Matrix](matrix/trait.Matrix.html), [SquareMatrix](square/trait.SquareMatrix.html), and
//! [HermiteMatrix](hermite/trait.HermiteMatrix.html)
//!
//! Matrix
//! -------
//! - [singular-value decomposition](matrix/trait.Matrix.html#tymethod.svd)
//! - [QR decomposition](matrix/trait.Matrix.html#tymethod.qr)
//! - [operator norm for L1 norm](matrix/trait.Matrix.html#tymethod.norm_1)
//! - [operator norm for L-inf norm](matrix/trait.Matrix.html#tymethod.norm_i)
//! - [Frobeiuns norm](matrix/trait.Matrix.html#tymethod.norm_f)
//! - [LU factorization](matrix/trait.Matrix.html#tymethod.lu)
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

extern crate lapack;
extern crate num_traits;
#[macro_use(s)]
extern crate ndarray;

pub mod prelude;
pub mod error;
pub mod vector;
pub mod matrix;
pub mod square;
pub mod hermite;
pub mod triangular;

pub mod qr;
pub mod svd;
pub mod eigh;
pub mod norm;
pub mod solve;
pub mod cholesky;
