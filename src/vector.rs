//! Define trait for vectors

use std::iter::Sum;
use ndarray::{LinalgScalar, ArrayBase, Data, Dimension};
use num_traits::float::Float;

/// Methods for vectors
pub trait Vector {
    type Scalar;
    /// rename of norm_l2
    fn norm(&self) -> Self::Scalar {
        self.norm_l2()
    }
    /// L-1 norm
    fn norm_l1(&self) -> Self::Scalar;
    /// L-2 norm
    fn norm_l2(&self) -> Self::Scalar;
    /// maximum norm
    fn norm_max(&self) -> Self::Scalar;
}

impl<A, S, D, T> Vector for ArrayBase<S, D>
    where A: LinalgScalar + Squared<Output = T>,
          T: Float + Sum,
          S: Data<Elem = A>,
          D: Dimension
{
    type Scalar = T;
    fn norm_l1(&self) -> Self::Scalar {
        self.iter().map(|x| x.sq_abs()).sum()
    }
    fn norm_l2(&self) -> Self::Scalar {
        self.iter().map(|x| x.squared()).sum::<T>().sqrt()
    }
    fn norm_max(&self) -> Self::Scalar {
        self.iter().fold(T::zero(), |f, &val| {
            let v = val.sq_abs();
            if f > v { f } else { v }
        })
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
