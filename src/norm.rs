//! Norm of vectors

use ndarray::*;
use num_traits::Zero;

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

impl<A, S, D> Norm for ArrayBase<S, D>
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
    D: Dimension,
{
    type Output = A::Real;
    fn norm_l1(&self) -> Self::Output {
        self.iter().map(|x| x.abs()).sum()
    }
    fn norm_l2(&self) -> Self::Output {
        self.iter().map(|x| x.square()).sum::<A::Real>().sqrt()
    }
    fn norm_max(&self) -> Self::Output {
        self.iter().fold(A::Real::zero(), |f, &val| {
            let v = val.abs();
            if f > v {
                f
            } else {
                v
            }
        })
    }
}

pub enum NormalizeAxis {
    Row = 0,
    Column = 1,
}

/// normalize in L2 norm
pub fn normalize<A, S>(
    mut m: ArrayBase<S, Ix2>,
    axis: NormalizeAxis,
) -> (ArrayBase<S, Ix2>, Vec<A::Real>)
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
{
    let mut ms = Vec::new();
    for mut v in m.axis_iter_mut(Axis(axis as usize)) {
        let n = v.norm();
        ms.push(n);
        v.map_inplace(|x| *x = *x / A::from_real(n))
    }
    (m, ms)
}
