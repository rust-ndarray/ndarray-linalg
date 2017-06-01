
use ndarray::*;

use super::error::*;

pub enum Layout {
    C((usize, usize)),
    F((usize, usize)),
}

impl Layout {
    pub fn size(&self) -> (usize, usize) {
        match self {
            &Layout::C(s) => s,
            &Layout::F(s) => s,
        }
    }
}

pub trait AllocatedArray2D {
    type Scalar;
    fn layout(&self) -> LResult<Layout>;
    fn square_layout(&self) -> LResult<Layout>;
    fn as_allocated(&self) -> LResult<&[Self::Scalar]>;
}

impl<A, S> AllocatedArray2D for ArrayBase<S, Ix2>
    where S: Data<Elem = A>
{
    type Scalar = A;

    fn layout(&self) -> LResult<Layout> {
        let strides = self.strides();
        if ::std::cmp::min(strides[0], strides[1]) != 1 {
            return Err(StrideError::new(strides[0], strides[1]).into());
        }
        if strides[0] < strides[1] {
            Ok(Layout::C((self.rows(), self.cols())))
        } else {
            Ok(Layout::F((self.rows(), self.cols())))
        }
    }

    fn square_layout(&self) -> LResult<Layout> {
        let l = self.layout()?;
        let (n, m) = l.size();
        if n == m {
            Ok(l)
        } else {
            Err(NotSquareError::new(n, m).into())
        }
    }

    fn as_allocated(&self) -> LResult<&[A]> {
        let slice = self.as_slice_memory_order().ok_or(MemoryContError::new())?;
        Ok(slice)
    }
}
