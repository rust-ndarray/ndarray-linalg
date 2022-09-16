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

/// Generate random array with given shape
///
/// - This function uses [rand::thread_rng].
///   See [random_using] for using another RNG
pub fn random<A, S, Sh, D>(sh: Sh) -> ArrayBase<S, D>
where
    A: Scalar,
    S: DataOwned<Elem = A>,
    D: Dimension,
    Sh: ShapeBuilder<Dim = D>,
{
    let mut rng = thread_rng();
    random_using(sh, &mut rng)
}

/// Generate random array with given RNG
///
/// - See [random] for using default RNG
pub fn random_using<A, S, Sh, D, R>(sh: Sh, rng: &mut R) -> ArrayBase<S, D>
where
    A: Scalar,
    S: DataOwned<Elem = A>,
    D: Dimension,
    Sh: ShapeBuilder<Dim = D>,
    R: Rng,
{
    ArrayBase::from_shape_fn(sh, |_| A::rand(rng))
}

/// Generate random unitary matrix using QR decomposition
///
/// - Be sure that this it **NOT** a uniform distribution.
///   Use it only for test purpose.
/// - This function uses [rand::thread_rng].
///   See [random_unitary_using] for using another RNG.
pub fn random_unitary<A>(n: usize) -> Array2<A>
where
    A: Scalar + Lapack,
{
    let mut rng = thread_rng();
    random_unitary_using(n, &mut rng)
}

/// Generate random unitary matrix using QR decomposition with given RNG
///
/// - Be sure that this it **NOT** a uniform distribution.
///   Use it only for test purpose.
/// - See [random_unitary] for using default RNG.
pub fn random_unitary_using<A, R>(n: usize, rng: &mut R) -> Array2<A>
where
    A: Scalar + Lapack,
    R: Rng,
{
    let a: Array2<A> = random_using((n, n), rng);
    let (q, _r) = a.qr_into().unwrap();
    q
}

/// Generate random regular matrix
///
/// - Be sure that this it **NOT** a uniform distribution.
///   Use it only for test purpose.
/// - This function uses [rand::thread_rng].
///   See [random_regular_using] for using another RNG.
pub fn random_regular<A>(n: usize) -> Array2<A>
where
    A: Scalar + Lapack,
{
    let mut rng = rand::thread_rng();
    random_regular_using(n, &mut rng)
}

/// Generate random regular matrix with given RNG
///
/// - Be sure that this it **NOT** a uniform distribution.
///   Use it only for test purpose.
/// - See [random_regular] for using default RNG.
pub fn random_regular_using<A, R>(n: usize, rng: &mut R) -> Array2<A>
where
    A: Scalar + Lapack,
    R: Rng,
{
    let a: Array2<A> = random_using((n, n), rng);
    let (q, mut r) = a.qr_into().unwrap();
    for i in 0..n {
        r[(i, i)] = A::one() + A::from_real(r[(i, i)].abs());
    }
    q.dot(&r)
}

/// Random Hermite matrix
///
/// - This function uses [rand::thread_rng].
///   See [random_hermite_using] for using another RNG.
pub fn random_hermite<A, S>(n: usize) -> ArrayBase<S, Ix2>
where
    A: Scalar,
    S: DataOwned<Elem = A> + DataMut,
{
    let mut rng = rand::thread_rng();
    random_hermite_using(n, &mut rng)
}

/// Random Hermite matrix with given RNG
///
/// - See [random_hermite] for using default RNG.
pub fn random_hermite_using<A, S, R>(n: usize, rng: &mut R) -> ArrayBase<S, Ix2>
where
    A: Scalar,
    S: DataOwned<Elem = A> + DataMut,
    R: Rng,
{
    let mut a: ArrayBase<S, Ix2> = random_using((n, n), rng);
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
/// - This function uses [rand::thread_rng].
///   See [random_hpd_using] for using another RNG.
///
pub fn random_hpd<A, S>(n: usize) -> ArrayBase<S, Ix2>
where
    A: Scalar,
    S: DataOwned<Elem = A> + DataMut,
{
    let mut rng = rand::thread_rng();
    random_hpd_using(n, &mut rng)
}

/// Random Hermite Positive-definite matrix with given RNG
///
/// - Eigenvalue of matrix must be larger than 1 (thus non-singular)
/// - See [random_hpd] for using default RNG.
///
pub fn random_hpd_using<A, S, R>(n: usize, rng: &mut R) -> ArrayBase<S, Ix2>
where
    A: Scalar,
    S: DataOwned<Elem = A> + DataMut,
    R: Rng,
{
    let a: Array2<A> = random_using((n, n), rng);
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
    let views: Vec<_> = xs.iter().map(|x| x.view()).collect();
    stack(Axis(1), &views).map_err(Into::into)
}

/// stack vectors into matrix vertically
pub fn vstack<A, S>(xs: &[ArrayBase<S, Ix1>]) -> Result<Array<A, Ix2>>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    let views: Vec<_> = xs.iter().map(|x| x.view()).collect();
    stack(Axis(0), &views).map_err(Into::into)
}
