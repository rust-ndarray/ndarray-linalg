//! Define trait for vectors

use std::iter::Sum;
use ndarray::{Array, NdFloat, Ix1, Ix2, LinalgScalar, ArrayBase, Data, Dimension};
use num_traits::float::Float;
use super::impls::outer::ImplOuter;

/// Norms of ndarray
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
    where A: LinalgScalar + NormedField<Output = T>,
          T: Float + Sum,
          S: Data<Elem = A>,
          D: Dimension
{
    type Output = T;
    fn norm_l1(&self) -> Self::Output {
        self.iter().map(|x| x.norm()).sum()
    }
    fn norm_l2(&self) -> Self::Output {
        self.iter().map(|x| x.squared()).sum::<T>().sqrt()
    }
    fn norm_max(&self) -> Self::Output {
        self.iter().fold(T::zero(), |f, &val| {
            let v = val.norm();
            if f > v { f } else { v }
        })
    }
}

/// Field with norm
pub trait NormedField {
    type Output: Float;
    fn squared(&self) -> Self::Output;
    fn norm(&self) -> Self::Output {
        self.squared().sqrt()
    }
}

impl<A: Float> NormedField for A {
    type Output = A;
    fn squared(&self) -> A {
        *self * *self
    }
    fn norm(&self) -> A {
        self.abs()
    }
}

/// Outer product
pub fn outer<A, S1, S2>(a: &ArrayBase<S1, Ix1>, b: &ArrayBase<S2, Ix1>) -> Array<A, Ix2>
    where A: NdFloat + ImplOuter,
          S1: Data<Elem = A>,
          S2: Data<Elem = A>
{
    let m = a.len();
    let n = b.len();
    let mut ab = Array::zeros((n, m));
    ImplOuter::outer(m,
                     n,
                     a.as_slice_memory_order().unwrap(),
                     b.as_slice_memory_order().unwrap(),
                     ab.as_slice_memory_order_mut().unwrap());
    ab.reversed_axes()
}
