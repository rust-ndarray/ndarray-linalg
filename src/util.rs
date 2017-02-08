//! misc utilities

use std::iter::Sum;
use ndarray::*;
use num_traits::Float;
use super::vector::*;
use std::ops::Div;

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
pub fn hstack<A, S>(xs: &[ArrayBase<S, Ix1>]) -> Result<Array<A, Ix2>, ShapeError>
    where A: LinalgScalar,
          S: Data<Elem = A>
{
    let views: Vec<_> = xs.iter()
        .map(|x| {
            let n = x.len();
            x.view().into_shape((n, 1)).unwrap()
        })
        .collect();
    stack(Axis(1), &views)
}

/// stack vectors into matrix vertically
pub fn vstack<A, S>(xs: &[ArrayBase<S, Ix1>]) -> Result<Array<A, Ix2>, ShapeError>
    where A: LinalgScalar,
          S: Data<Elem = A>
{
    let views: Vec<_> = xs.iter()
        .map(|x| {
            let n = x.len();
            x.view().into_shape((1, n)).unwrap()
        })
        .collect();
    stack(Axis(0), &views)
}

pub enum NormalizeAxis {
    Row = 0,
    Column = 1,
}

/// normalize in L2 norm
pub fn normalize<A, S, T>(mut m: ArrayBase<S, Ix2>, axis: NormalizeAxis) -> (ArrayBase<S, Ix2>, Vec<T>)
    where A: LinalgScalar + NormedField<Output = T> + Div<T, Output = A>,
          S: DataMut<Elem = A>,
          T: Float + Sum
{
    let mut ms = Vec::new();
    for mut v in m.axis_iter_mut(Axis(axis as usize)) {
        let n = v.norm();
        ms.push(n);
        v.map_inplace(|x| *x = *x / n)
    }
    (m, ms)
}

/// check two arrays are close in maximum norm
pub fn all_close_max<A, Tol, S1, S2, D>(test: &ArrayBase<S1, D>,
                                        truth: &ArrayBase<S2, D>,
                                        atol: Tol)
                                        -> Result<Tol, Tol>
    where A: LinalgScalar + NormedField<Output = Tol>,
          Tol: Float + Sum,
          S1: Data<Elem = A>,
          S2: Data<Elem = A>,
          D: Dimension
{
    let tol = (test - truth).norm_max();
    if tol < atol { Ok(tol) } else { Err(tol) }
}

/// check two arrays are close in L1 norm
pub fn all_close_l1<A, Tol, S1, S2, D>(test: &ArrayBase<S1, D>, truth: &ArrayBase<S2, D>, rtol: Tol) -> Result<Tol, Tol>
    where A: LinalgScalar + NormedField<Output = Tol>,
          Tol: Float + Sum,
          S1: Data<Elem = A>,
          S2: Data<Elem = A>,
          D: Dimension
{
    let tol = (test - truth).norm_l1() / truth.norm_l1();
    if tol < rtol { Ok(tol) } else { Err(tol) }
}

/// check two arrays are close in L2 norm
pub fn all_close_l2<A, Tol, S1, S2, D>(test: &ArrayBase<S1, D>, truth: &ArrayBase<S2, D>, rtol: Tol) -> Result<Tol, Tol>
    where A: LinalgScalar + NormedField<Output = Tol>,
          Tol: Float + Sum,
          S1: Data<Elem = A>,
          S2: Data<Elem = A>,
          D: Dimension
{
    let tol = (test - truth).norm_l2() / truth.norm_l2();
    if tol < rtol { Ok(tol) } else { Err(tol) }
}
