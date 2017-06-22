
use ndarray::*;
use std::ops::*;
use rand::*;

use super::layout::*;
use super::types::*;
use super::error::*;

pub fn conjugate<A, Si, So>(a: &ArrayBase<Si, Ix2>) -> ArrayBase<So, Ix2>
    where A: Conjugate,
          Si: Data<Elem = A>,
          So: DataOwned<Elem = A> + DataMut
{
    let mut a = replicate(&a.t());
    for val in a.iter_mut() {
        *val = Conjugate::conj(*val);
    }
    a
}

pub fn random<A, S, Sh, D>(sh: Sh) -> ArrayBase<S, D>
    where A: RandNormal,
          S: DataOwned<Elem = A>,
          D: Dimension,
          Sh: ShapeBuilder<Dim = D>
{
    let mut rng = thread_rng();
    ArrayBase::from_shape_fn(sh, |_| A::randn(&mut rng))
}

/// Random vector
pub fn random_vector<A, S>(n: usize) -> ArrayBase<S, Ix1>
    where A: RandNormal,
          S: DataOwned<Elem = A>
{
    let mut rng = thread_rng();
    let v: Vec<A> = (0..n).map(|_| A::randn(&mut rng)).collect();
    ArrayBase::from_vec(v)
}

/// Random matrix
pub fn random_matrix<A, S>(n: usize, m: usize) -> ArrayBase<S, Ix2>
    where A: RandNormal,
          S: DataOwned<Elem = A>
{
    let mut rng = thread_rng();
    let v: Vec<A> = (0..n * m).map(|_| A::randn(&mut rng)).collect();
    ArrayBase::from_shape_vec((n, m), v).unwrap()
}

/// Random square matrix
pub fn random_square<A, S>(n: usize) -> ArrayBase<S, Ix2>
    where A: RandNormal,
          S: DataOwned<Elem = A>
{
    random_matrix(n, n)
}

/// Random Hermite matrix
pub fn random_hermite<A, S>(n: usize) -> ArrayBase<S, Ix2>
    where A: RandNormal + Conjugate + Add<Output = A>,
          S: DataOwned<Elem = A> + DataMut
{
    let mut a = random_square(n);
    for i in 0..n {
        a[(i, i)] = a[(i, i)] + Conjugate::conj(a[(i, i)]);
        for j in (i + 1)..n {
            a[(i, j)] = Conjugate::conj(a[(j, i)])
        }
    }
    a
}

/// Random Hermite Positive-definite matrix
pub fn random_hpd<A, S>(n: usize) -> ArrayBase<S, Ix2>
    where A: RandNormal + Conjugate + LinalgScalar,
          S: DataOwned<Elem = A> + DataMut
{
    let a: Array2<A> = random_square(n);
    let ah: Array2<A> = conjugate(&a);
    replicate(&ah.dot(&a))
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
