//! module for topologcal vector space
//!

use std::iter::Sum;
use ndarray::{ArrayBase, Data, Dimension, LinalgScalar, IntoDimension};
use num_traits::Float;

pub fn inf<A, Distance, S1, S2, D>(a: &ArrayBase<S1, D>, b: &ArrayBase<S2, D>) -> Result<Distance, String>
    where A: LinalgScalar + Squared<Output = Distance>,
          Distance: Float + Sum,
          S1: Data<Elem = A>,
          S2: Data<Elem = A>,
          D: Dimension
{
    if a.shape() != b.shape() {
        return Err("Shapes are different".into());
    }
    let mut max_tol = Distance::zero();
    for (idx, val) in a.indexed_iter() {
        let t = b[idx.into_dimension()];
        let tol = (*val - t).sq_abs();
        if tol > max_tol {
            max_tol = tol;
        }
    }
    Ok(max_tol)
}

pub fn l1<A, Distance, S1, S2, D>(a: &ArrayBase<S1, D>, b: &ArrayBase<S2, D>) -> Result<Distance, String>
    where A: LinalgScalar + Squared<Output = Distance>,
          Distance: Float + Sum,
          S1: Data<Elem = A>,
          S2: Data<Elem = A>,
          D: Dimension
{
    if a.shape() != b.shape() {
        return Err("Shapes are different".into());
    }
    Ok(a.indexed_iter().map(|(idx, val)| (b[idx.into_dimension()] - *val).sq_abs()).sum())
}

pub fn l2<A, Distance, S1, S2, D>(a: &ArrayBase<S1, D>, b: &ArrayBase<S2, D>) -> Result<Distance, String>
    where A: LinalgScalar + Squared<Output = Distance>,
          Distance: Float + Sum,
          S1: Data<Elem = A>,
          S2: Data<Elem = A>,
          D: Dimension
{
    if a.shape() != b.shape() {
        return Err("Shapes are different".into());
    }
    Ok(a.indexed_iter().map(|(idx, val)| (b[idx.into_dimension()] - *val).squared()).sum::<Distance>().sqrt())
}

#[derive(Debug)]
pub enum NotCloseError<Tol> {
    ShapeMismatch(String),
    LargeDeviation(Tol),
}

pub fn all_close_inf<A, Tol, S1, S2, D>(a: &ArrayBase<S1, D>,
                                        b: &ArrayBase<S2, D>,
                                        atol: Tol)
                                        -> Result<Tol, NotCloseError<Tol>>
    where A: LinalgScalar + Squared<Output = Tol>,
          Tol: Float + Sum,
          S1: Data<Elem = A>,
          S2: Data<Elem = A>,
          D: Dimension
{
    if a.shape() != b.shape() {
        return Err(NotCloseError::ShapeMismatch("Shapes are different".into()));
    }
    let mut max_tol = Tol::zero();
    for (idx, val) in a.indexed_iter() {
        let t = b[idx.into_dimension()];
        let tol = (*val - t).sq_abs();
        if tol > atol {
            return Err(NotCloseError::LargeDeviation(tol));
        }
        if tol > max_tol {
            max_tol = tol;
        }
    }
    Ok(max_tol)
}

pub fn all_close_l1<A, Tol, S1, S2, D>(a: &ArrayBase<S1, D>,
                                       b: &ArrayBase<S2, D>,
                                       rtol: Tol)
                                       -> Result<Tol, NotCloseError<Tol>>
    where A: LinalgScalar + Squared<Output = Tol>,
          Tol: Float + Sum,
          S1: Data<Elem = A>,
          S2: Data<Elem = A>,
          D: Dimension
{
    if a.shape() != b.shape() {
        return Err(NotCloseError::ShapeMismatch("Shapes are different".into()));
    }
    let nrm: Tol = b.iter().map(|x| x.sq_abs()).sum();
    let dev: Tol = a.indexed_iter().map(|(idx, val)| (b[idx.into_dimension()] - *val).sq_abs()).sum();
    if dev / nrm > rtol {
        Err(NotCloseError::LargeDeviation(dev / nrm))
    } else {
        Ok(dev / nrm)
    }
}

pub fn all_close_l2<A, Tol, S1, S2, D>(a: &ArrayBase<S1, D>,
                                       b: &ArrayBase<S2, D>,
                                       rtol: Tol)
                                       -> Result<Tol, NotCloseError<Tol>>
    where A: LinalgScalar + Squared<Output = Tol>,
          Tol: Float + Sum,
          S1: Data<Elem = A>,
          S2: Data<Elem = A>,
          D: Dimension
{
    if a.shape() != b.shape() {
        return Err(NotCloseError::ShapeMismatch("Shapes are different".into()));
    }
    let nrm: Tol = b.iter().map(|x| x.squared()).sum();
    let dev: Tol = a.indexed_iter().map(|(idx, val)| (b[idx.into_dimension()] - *val).squared()).sum();
    let d = (dev / nrm).sqrt();
    if d > rtol {
        Err(NotCloseError::LargeDeviation(d))
    } else {
        Ok(d)
    }
}

pub trait Squared {
    type Output;
    fn squared(&self) -> Self::Output;
    fn sq_abs(&self) -> Self::Output;
}

impl<A: Float> Squared for A {
    type Output = A;
    fn squared(&self) -> A {
        *self * *self
    }
    fn sq_abs(&self) -> A {
        self.abs()
    }
}
