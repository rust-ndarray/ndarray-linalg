//! Decomposition with LOBPCG
//!
//! Locally Optimal Block Preconditioned Conjugate Gradient (LOBPCG) is a matrix-free method for
//! finding the large (or smallest) eigenvalues and the corresponding eigenvectors of a symmetric
//! eigenvalue problem
//! ```text
//! A x = lambda x
//! ```
//! where A is symmetric and (x, lambda) the solution. It has the following advantages:
//! * matrix free: does not require storing the coefficient matrix explicitely and only evaluates
//! matrix-vector products.
//! * factorization-free: does not require any matrix decomposition
//! * linear-convergence: theoretically guaranteed and practically observed
//!
//! See also the wikipedia article at [LOBPCG](https://en.wikipedia.org/wiki/LOBPCG)
//!
mod eig;
mod lobpcg;
mod svd;

pub use eig::{TruncatedEig, TruncatedEigIterator};
pub use lobpcg::{lobpcg, LobpcgResult, Order as TruncatedOrder};
pub use svd::{MagnitudeCorrection, TruncatedSvd};
