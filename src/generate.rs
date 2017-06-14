
use ndarray::*;
use rand::*;
use std::ops::*;

use super::types::*;
use super::error::*;

pub fn random_square<A, S>(n: usize) -> ArrayBase<S, Ix2>
    where A: Rand,
          S: DataOwned<Elem = A>
{
    let mut rng = thread_rng();
    let v: Vec<A> = (0..n * n).map(|_| rng.gen()).collect();
    ArrayBase::from_shape_vec((n, n), v).unwrap()
}

pub fn random_hermite<A, S>(n: usize) -> ArrayBase<S, Ix2>
    where A: Rand + Conjugate + Add<Output = A>,
          S: DataOwned<Elem = A> + DataMut
{
    let mut a = random_square(n);
    for i in 0..n {
        a[(i, i)] = a[(i, i)] + Conjugate::conj(a[(i, i)]);
        for j in i..n {
            a[(i, j)] = a[(j, i)];
        }
    }
    a
}

/// construct matrix from diag
pub fn from_diag<A>(d: &[A]) -> Array2<A>
    where A: LinalgScalar
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
    where A: LinalgScalar,
          S: Data<Elem = A>
{
    let views: Vec<_> = xs.iter()
        .map(|x| {
            let n = x.len();
            x.view().into_shape((n, 1)).unwrap()
        })
        .collect();
    stack(Axis(1), &views).map_err(|e| e.into())
}

/// stack vectors into matrix vertically
pub fn vstack<A, S>(xs: &[ArrayBase<S, Ix1>]) -> Result<Array<A, Ix2>>
    where A: LinalgScalar,
          S: Data<Elem = A>
{
    let views: Vec<_> = xs.iter()
        .map(|x| {
            let n = x.len();
            x.view().into_shape((1, n)).unwrap()
        })
        .collect();
    stack(Axis(0), &views).map_err(|e| e.into())
}
