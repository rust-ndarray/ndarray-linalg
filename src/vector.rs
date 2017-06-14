//! Define trait for vectors

use std::iter::Sum;
use ndarray::*;
use num_traits::Float;

use super::types::*;

/// Define norm as a metric linear space (not as a matrix)
///
/// For operator norms, see opnorm module
pub trait Norm {
    type Output;
    /// rename of `norm_l2`
    fn norm(&self) -> Self::Output {
        self.norm_l2()
    }
    /// L-1 norm
    fn norm_l1(&self) -> Self::Output;
    /// L-2 norm
    fn norm_l2(&self) -> Self::Output;
    /// maximum norm
    fn norm_max(&self) -> Self::Output;
}

impl<A, S, D, T> Norm for ArrayBase<S, D>
    where A: LinalgScalar + Absolute<Output = T>,
          T: Float + Sum,
          S: Data<Elem = A>,
          D: Dimension
{
    type Output = T;
    fn norm_l1(&self) -> Self::Output {
        self.iter().map(|x| x.abs()).sum()
    }
    fn norm_l2(&self) -> Self::Output {
        self.iter().map(|x| x.squared()).sum::<T>().sqrt()
    }
    fn norm_max(&self) -> Self::Output {
        self.iter().fold(T::zero(), |f, &val| {
            let v = val.abs();
            if f > v { f } else { v }
        })
    }
}
