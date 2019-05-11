use crate::types::*;
use ndarray::*;

mod householder;
mod mgs;

pub use householder::Householder;
pub use mgs::MGS;

/// Q-matrix (unitary)
pub type Q<A> = Array2<A>;
/// R-matrix (upper triangle)
pub type R<A> = Array2<A>;

pub trait Orthogonalizer {
    type Elem: Scalar;

    fn dim(&self) -> usize;
    fn len(&self) -> usize;

    /// check if the basis spans entire space
    fn is_full(&self) -> bool {
        self.len() == self.dim()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Orthogonalize given vector
    ///
    /// Returns
    /// --------
    /// Residual norm
    ///
    /// Panic
    /// -------
    /// - if the size of the input array mismaches to the dimension
    ///
    fn orthogonalize<S>(&self, a: &mut ArrayBase<S, Ix1>) -> <Self::Elem as Scalar>::Real
    where
        S: DataMut<Elem = Self::Elem>;

    /// Add new vector if the residual is larger than relative tolerance
    ///
    /// Returns
    /// --------
    /// Coefficients to the `i`-th Q-vector
    ///
    /// - The size of array must be `self.len() + 1`
    /// - The last element is the residual norm of input vector
    ///
    /// Panic
    /// -------
    /// - if the size of the input array mismaches to the dimension
    ///
    fn append<S>(
        &mut self,
        a: ArrayBase<S, Ix1>,
        rtol: <Self::Elem as Scalar>::Real,
    ) -> Result<Array1<Self::Elem>, Array1<Self::Elem>>
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

    /// Orghotonalize dependent vector without adding to Q,
    /// thus R must be non-regular like following:
    ///
    /// ```ignore
    /// x x x x x
    /// 0 x x x x
    /// 0 0 0 x x
    /// 0 0 0 0 x
    /// 0 0 0 0 0   // 0-filled to be square matrix
    /// ```
    Full,
}

/// Online QR decomposition using arbitary orthogonalizer
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

/// Online QR decomposition using modified Gram-Schmit
pub fn mgs<A, S>(
    iter: impl Iterator<Item = ArrayBase<S, Ix1>>,
    dim: usize,
    rtol: A::Real,
    strategy: Strategy,
) -> (Q<A>, R<A>)
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
{
    let mgs = MGS::new(dim);
    qr(iter, mgs, rtol, strategy)
}

/// Online QR decomposition using modified Gram-Schmit
pub fn householder<A, S>(
    iter: impl Iterator<Item = ArrayBase<S, Ix1>>,
    dim: usize,
    rtol: A::Real,
    strategy: Strategy,
) -> (Q<A>, R<A>)
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
{
    let h = Householder::new(dim);
    qr(iter, h, rtol, strategy)
}
