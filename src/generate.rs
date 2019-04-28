//! Generator functions for matrices

use ndarray::*;
use rand::prelude::*;

use super::convert::*;
use super::error::*;
use super::qr::*;
use super::types::*;

/// Hermite conjugate matrix
pub fn conjugate<A, Si, So>(a: &ArrayBase<Si, Ix2>) -> ArrayBase<So, Ix2>
where
    A: Scalar,
    Si: Data<Elem = A>,
    So: DataOwned<Elem = A> + DataMut,
{
    let mut a: ArrayBase<So, Ix2> = replicate(&a.t());
    for val in a.iter_mut() {
        *val = val.conj();
    }
    a
}

/// Generate random array
pub fn random<A, S, Sh, D>(sh: Sh) -> ArrayBase<S, D>
where
    A: Scalar,
    S: DataOwned<Elem = A>,
    D: Dimension,
    Sh: ShapeBuilder<Dim = D>,
{
    let mut rng = thread_rng();
    ArrayBase::from_shape_fn(sh, |_| A::rand(&mut rng))
}

/// Generate random unitary matrix using QR decomposition
///
/// Be sure that this it **NOT** a uniform distribution. Use it only for test purpose.
pub fn random_unitary<A>(n: usize) -> Array2<A>
where
    A: Scalar + Lapack,
{
    let a: Array2<A> = random((n, n));
    let (q, _r) = a.qr_into().unwrap();
    q
}

/// Generate random regular matrix
///
/// Be sure that this it **NOT** a uniform distribution. Use it only for test purpose.
pub fn random_regular<A>(n: usize) -> Array2<A>
where
    A: Scalar + Lapack,
{
    let a: Array2<A> = random((n, n));
    let (q, mut r) = a.qr_into().unwrap();
    for i in 0..n {
        r[(i, i)] = A::one() + A::from_real(r[(i, i)].abs());
    }
    q.dot(&r)
}

/// Random Hermite matrix
pub fn random_hermite<A, S>(n: usize) -> ArrayBase<S, Ix2>
where
    A: Scalar,
    S: DataOwned<Elem = A> + DataMut,
{
    let mut a: ArrayBase<S, Ix2> = random((n, n));
    for i in 0..n {
        a[(i, i)] = a[(i, i)] + a[(i, i)].conj();
        for j in (i + 1)..n {
            a[(i, j)] = a[(j, i)].conj();
        }
    }
    a
}

/// Random Hermite Positive-definite matrix
///
/// - Eigenvalue of matrix must be larger than 1 (thus non-singular)
///
pub fn random_hpd<A, S>(n: usize) -> ArrayBase<S, Ix2>
where
    A: Scalar,
    S: DataOwned<Elem = A> + DataMut,
{
    let a: Array2<A> = random((n, n));
    let ah: Array2<A> = conjugate(&a);
    ArrayBase::eye(n) + &ah.dot(&a)
}

/// construct matrix from diag
pub fn from_diag<A>(d: &[A]) -> Array2<A>
where
    A: Scalar,
{
    let n = d.len();
    let mut e = Array::zeros((n, n));
    for i in 0..n {
        e[(i, i)] = d[i];
    }
    e
}

/// stack vectors into matrix horizontally
pub fn hstack<A, S>(xs: &[ArrayBase<S, Ix1>]) -> Result<Array<A, Ix2>>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    let views: Vec<_> = xs
        .iter()
        .map(|x| {
            let n = x.len();
            x.view().into_shape((n, 1)).unwrap()
        })
        .collect();
    stack(Axis(1), &views).map_err(|e| e.into())
}

/// stack vectors into matrix vertically
pub fn vstack<A, S>(xs: &[ArrayBase<S, Ix1>]) -> Result<Array<A, Ix2>>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    let views: Vec<_> = xs
        .iter()
        .map(|x| {
            let n = x.len();
            x.view().into_shape((1, n)).unwrap()
        })
        .collect();
    stack(Axis(0), &views).map_err(|e| e.into())
}
