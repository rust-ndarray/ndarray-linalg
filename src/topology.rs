//! module for topologcal vector space
//!

use std::iter::Sum;
use ndarray::{ArrayBase, Data, Dimension, LinalgScalar};
use num_traits::Float;
use super::vector::*;

pub fn all_close_max<A, Tol, S1, S2, D>(test: &ArrayBase<S1, D>,
                                        truth: &ArrayBase<S2, D>,
                                        atol: Tol)
                                        -> Result<Tol, Tol>
    where A: LinalgScalar + Squared<Output = Tol>,
          Tol: Float + Sum,
          S1: Data<Elem = A>,
          S2: Data<Elem = A>,
          D: Dimension
{
    let tol = (test - truth).norm_max();
    if tol < atol { Ok(tol) } else { Err(tol) }
}

pub fn all_close_l1<A, Tol, S1, S2, D>(test: &ArrayBase<S1, D>, truth: &ArrayBase<S2, D>, rtol: Tol) -> Result<Tol, Tol>
    where A: LinalgScalar + Squared<Output = Tol>,
          Tol: Float + Sum,
          S1: Data<Elem = A>,
          S2: Data<Elem = A>,
          D: Dimension
{
    let tol = (test - truth).norm_l1() / truth.norm_l1();
    if tol < rtol { Ok(tol) } else { Err(tol) }
}

pub fn all_close_l2<A, Tol, S1, S2, D>(test: &ArrayBase<S1, D>, truth: &ArrayBase<S2, D>, rtol: Tol) -> Result<Tol, Tol>
    where A: LinalgScalar + Squared<Output = Tol>,
          Tol: Float + Sum,
          S1: Data<Elem = A>,
          S2: Data<Elem = A>,
          D: Dimension
{
    let tol = (test - truth).norm_l2() / truth.norm_l2();
    if tol < rtol { Ok(tol) } else { Err(tol) }
}
