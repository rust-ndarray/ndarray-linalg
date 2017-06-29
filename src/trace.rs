//! Trace calculation

use ndarray::*;
use std::iter::Sum;

use super::error::*;
use super::layout::*;
use super::types::*;

pub trait Trace {
    type Output;
    fn trace(&self) -> Result<Self::Output>;
}

impl<A, S> Trace for ArrayBase<S, Ix2>
where
    A: Scalar + Sum,
    S: Data<Elem = A>,
{
    type Output = A;

    fn trace(&self) -> Result<Self::Output> {
        let (n, _) = self.square_layout()?.size();
        Ok((0..n as usize).map(|i| self[(i, i)]).sum())
    }
}
