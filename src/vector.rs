//! Define trait for vectors

use ndarray::{NdFloat, Array, RcArray, Ix1};

/// Methods for vectors
pub trait Vector {
    type Scalar;
    /// L-2 norm
    fn norm(&self) -> Self::Scalar;
}

impl<A: NdFloat> Vector for Array<A, Ix1> {
    type Scalar = A;
    fn norm(&self) -> Self::Scalar {
        self.dot(&self).sqrt()
    }
}

impl<A: NdFloat> Vector for RcArray<A, Ix1> {
    type Scalar = A;
    fn norm(&self) -> Self::Scalar {
        self.dot(&self).sqrt()
    }
}
