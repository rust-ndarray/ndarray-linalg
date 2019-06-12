//! Krylov subspace

use crate::types::*;
use ndarray::*;

pub mod householder;
pub mod mgs;

pub use householder::{householder, Householder};
pub use mgs::{mgs, MGS};

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

/// Array type for coefficients to the current basis
///
/// - The length must be `self.len() + 1`
/// - Last component is the residual norm
///
pub type Coefficients<A> = Array1<A>;

/// Trait for creating orthogonal basis from iterator of arrays
///
/// Panic
/// -------
/// - if the size of the input array mismatches to the dimension
///
/// Example
/// -------
///
/// ```rust
/// # use ndarray::*;
/// # use ndarray_linalg::{krylov::*, *};
/// let mut mgs = MGS::new(3);
/// let coef = mgs.append(array![0.0, 1.0, 0.0], 1e-9).unwrap();
/// close_l2(&coef, &array![1.0], 1e-9);
///
/// let coef = mgs.append(array![1.0, 1.0, 0.0], 1e-9).unwrap();
/// close_l2(&coef, &array![1.0, 1.0], 1e-9);
///
/// // Fail if the vector is linearly dependent
/// assert!(mgs.append(array![1.0, 2.0, 0.0], 1e-9).is_err());
///
/// // You can get coefficients of dependent vector
/// if let Err(coef) = mgs.append(array![1.0, 2.0, 0.0], 1e-9) {
///     close_l2(&coef, &array![2.0, 1.0, 0.0], 1e-9);
/// }
/// ```
pub trait Orthogonalizer {
    type Elem: Scalar;

    /// Dimension of input array
    fn dim(&self) -> usize;

    /// Number of cached basis
    fn len(&self) -> usize;

    /// check if the basis spans entire space
    fn is_full(&self) -> bool {
        self.len() == self.dim()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Decompose given vector into the span of current basis and
    /// its tangent space
    ///
    /// - `a` becomes the tangent vector
    /// - The Coefficients to the current basis is returned.
    ///
    fn decompose<S>(&self, a: &mut ArrayBase<S, Ix1>) -> Coefficients<Self::Elem>
    where
        S: DataMut<Elem = Self::Elem>;

    /// Calculate the coefficient to the current basis basis
    ///
    /// - This will be faster than `decompose` because the construction of the residual vector may
    ///   requires more Calculation
    ///
    fn coeff<S>(&self, a: ArrayBase<S, Ix1>) -> Coefficients<Self::Elem>
    where
        S: Data<Elem = Self::Elem>;

    /// Add new vector if the residual is larger than relative tolerance
    fn append<S>(
        &mut self,
        a: ArrayBase<S, Ix1>,
        rtol: <Self::Elem as Scalar>::Real,
    ) -> Result<Coefficients<Self::Elem>, Coefficients<Self::Elem>>
    where
        S: DataMut<Elem = Self::Elem>;

    /// Get Q-matrix of generated basis
    fn get_q(&self) -> Q<Self::Elem>;
}

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

/// Online QR decomposition using arbitrary orthogonalizer
pub fn qr<A, S>(
    iter: impl Iterator<Item = ArrayBase<S, Ix1>>,
    mut ortho: impl Orthogonalizer<Elem = A>,
    rtol: A::Real,
    strategy: Strategy,
) -> (Q<A>, R<A>)
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
{
    assert_eq!(ortho.len(), 0);

    let mut coefs = Vec::new();
    for a in iter {
        match ortho.append(a.into_owned(), rtol) {
            Ok(coef) => coefs.push(coef),
            Err(coef) => match strategy {
                Strategy::Terminate => break,
                Strategy::Skip => continue,
                Strategy::Full => coefs.push(coef),
            },
        }
    }
    let n = ortho.len();
    let m = coefs.len();
    let mut r = Array2::zeros((n, m).f());
    for j in 0..m {
        for i in 0..n {
            if i < coefs[j].len() {
                r[(i, j)] = coefs[j][i];
            }
        }
    }
    (ortho.get_q(), r)
}
