//! Krylov subspace

use crate::types::*;
use ndarray::*;

pub mod mgs;

pub use mgs::MGS;

/// Q-matrix
///
/// - Maybe **NOT** square
/// - Unitary for existing columns
///
pub type Q<A> = Array2<A>;

/// R-matrix
///
/// - Maybe **NOT** square
/// - Upper triangle
///
pub type R<A> = Array2<A>;

/// Strategy for linearly dependent vectors appearing in iterative QR decomposition
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Strategy {
    /// Terminate iteration if dependent vector comes
    Terminate,

    /// Skip dependent vector
    Skip,

    /// Orthogonalize dependent vector without adding to Q,
    /// i.e. R must be non-square like following:
    ///
    /// ```text
    /// x x x x x
    /// 0 x x x x
    /// 0 0 0 x x
    /// 0 0 0 0 x
    /// ```
    Full,
}
