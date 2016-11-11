//! Define trait for vectors

use ndarray::{LinalgScalar, Array, Ix1};
use num_traits::float::Float;

/// Methods for vectors
pub trait Vector {
    type Scalar;
    /// L-2 norm
    fn norm(&self) -> Self::Scalar;
}

impl<A: Float + LinalgScalar> Vector for Array<A, Ix1> {
    type Scalar = A;
    fn norm(&self) -> Self::Scalar {
        self.dot(&self).sqrt()
    }
}
