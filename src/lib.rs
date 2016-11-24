//! This crate serves matrix manipulation for
//! [rust-ndarray](https://github.com/bluss/rust-ndarray).
//!
//! They are implemented as traits,
//! [Matrix](matrix/trait.Matrix.html), [SquareMatrix](square/trait.SquareMatrix.html),
//! [TriangularMatrix](triangular/trait.TriangularMatrix.html), and
//! [HermiteMatrix](hermite/trait.HermiteMatrix.html)
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
//! TriangularMatrix
//! ------------------
//! - [solve linear problem with upper triangular matrix](triangular/trait.TriangularMatrix.html#tymethod.solve_upper)
//! - [solve linear problem with lower triangular matrix](triangular/trait.TriangularMatrix.html#tymethod.solve_lower)
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
