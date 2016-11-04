
use ndarray::prelude::*;
use ndarray::LinalgScalar;
use num_traits::float::Float;

pub trait Vector {
    type Scalar;
    fn norm(&self) -> Self::Scalar;
}

impl<A: Float + LinalgScalar> Vector for Array<A, Ix> {
    type Scalar = A;
    fn norm(&self) -> Self::Scalar {
        self.dot(&self).sqrt()
    }
}
