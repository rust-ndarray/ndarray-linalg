//! misc utilities

use std::iter::Sum;
use ndarray::*;
use num_traits::Float;
use super::vector::*;
use std::ops::Div;

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

/// normalize columns in L2 norm
pub fn normalize_columns<A, S, T>(m: &ArrayBase<S, Ix2>) -> (Array2<A>, Vec<T>)
    where S: Data<Elem = A>,
          A: LinalgScalar + NormedField<Output = T> + Div<T, Output = A>,
          T: Float + Sum
{
    let mut ms = Vec::new();
    let vs = m.axis_iter(Axis(1))
        .map(|v| {
            let n = v.norm();
            ms.push(n);
            v.mapv(|x| x / n)
        })
        .collect::<Vec<_>>();
    (hstack(&vs).unwrap(), ms)
}

/// normalize rows in L2 norm
pub fn normalize_rows<A, S, T>(m: &ArrayBase<S, Ix2>) -> (Vec<T>, Array2<A>)
    where S: Data<Elem = A>,
          A: LinalgScalar + NormedField<Output = T> + Div<T, Output = A>,
          T: Float + Sum
{
    let mut ms = Vec::new();
    let vs = m.axis_iter(Axis(0))
        .map(|v| {
            let n = v.norm();
            ms.push(n);
            v.mapv(|x| x / n)
        })
        .collect::<Vec<_>>();
    (ms, vstack(&vs).unwrap())
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
