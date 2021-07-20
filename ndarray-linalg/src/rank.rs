///! Computes the rank of a matrix using single value decomposition
use ndarray::*;

use super::error::*;
use super::svd::SVD;
use super::types::*;
use num_traits::Float;

pub trait Rank {
    fn rank(&self) -> Result<Ix>;
}

impl<A, S> Rank for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
{
    fn rank(&self) -> Result<Ix> {
        let (_, sv, _) = self.svd(false, false)?;

        let (n, m) = self.dim();
        let tol = A::Real::epsilon() * A::Real::real(n.max(m)) * sv[0];

        let output = sv.iter().take_while(|v| v > &&tol).count();
        Ok(output)
    }
}
