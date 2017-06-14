//! misc utilities

use ndarray::*;
use std::ops::Div;

use super::types::*;
use super::vector::*;

pub enum NormalizeAxis {
    Row = 0,
    Column = 1,
}

/// normalize in L2 norm
pub fn normalize<A, S, T>(mut m: ArrayBase<S, Ix2>, axis: NormalizeAxis) -> (ArrayBase<S, Ix2>, Vec<T>)
    where A: Field + Absolute<Output = T> + Div<T, Output = A>,
          S: DataMut<Elem = A>,
          T: RealField
{
    let mut ms = Vec::new();
    for mut v in m.axis_iter_mut(Axis(axis as usize)) {
        let n = v.norm();
        ms.push(n);
        v.map_inplace(|x| *x = *x / n)
    }
    (m, ms)
}
