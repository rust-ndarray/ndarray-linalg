
use std::iter::Sum;
use ndarray::*;
use num_traits::Float;
use super::vector::*;

/// stack vectors into matrix horizontally
pub fn hstack<A, S>(xs: &[ArrayBase<S, Ix1>]) -> Result<Array<A, Ix2>, ShapeError>
    where A: NdFloat,
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
    where A: NdFloat,
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
