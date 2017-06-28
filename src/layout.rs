//! Memory layout of matrices

use lapack::c;
use ndarray::*;

use super::error::*;

pub type LDA = i32;
pub type LEN = i32;
pub type Col = i32;
pub type Row = i32;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Layout {
    C((Row, LDA)),
    F((Col, LDA)),
}

impl Layout {
    pub fn size(&self) -> (Row, Col) {
        match *self {
            Layout::C((row, lda)) => (row, lda),
            Layout::F((col, lda)) => (lda, col),
        }
    }

    pub fn resized(&self, row: Row, col: Col) -> Layout {
        match *self {
            Layout::C(_) => Layout::C((row, col)),
            Layout::F(_) => Layout::F((col, row)),
        }
    }

    pub fn lda(&self) -> LDA {
        match *self {
            Layout::C((_, lda)) => lda,
            Layout::F((_, lda)) => lda,
        }
    }

    pub fn len(&self) -> LEN {
        match *self {
            Layout::C((row, _)) => row,
            Layout::F((col, _)) => col,
        }
    }

    pub fn lapacke_layout(&self) -> c::Layout {
        match *self {
            Layout::C(_) => c::Layout::RowMajor,
            Layout::F(_) => c::Layout::ColumnMajor,
        }
    }

    pub fn same_order(&self, other: &Layout) -> bool {
        self.lapacke_layout() == other.lapacke_layout()
    }

    pub fn as_shape(&self) -> Shape<Ix2> {
        match *self {
            Layout::C((row, col)) => (row as usize, col as usize).into_shape(),
            Layout::F((col, row)) => (row as usize, col as usize).f().into_shape(),
        }
    }

    pub fn toggle_order(&self) -> Self {
        match *self {
            Layout::C((row, col)) => Layout::F((col, row)),
            Layout::F((col, row)) => Layout::C((row, col)),
        }
    }
}

pub trait AllocatedArray {
    type Elem;
    fn layout(&self) -> Result<Layout>;
    fn square_layout(&self) -> Result<Layout>;
    fn as_allocated(&self) -> Result<&[Self::Elem]>;
}

pub trait AllocatedArrayMut: AllocatedArray {
    fn as_allocated_mut(&mut self) -> Result<&mut [Self::Elem]>;
}

impl<A, S> AllocatedArray for ArrayBase<S, Ix2>
where
    S: Data<Elem = A>,
{
    type Elem = A;

    fn layout(&self) -> Result<Layout> {
        let shape = self.shape();
        let strides = self.strides();
        if shape[0] == strides[1] as usize {
            return Ok(Layout::F((self.cols() as i32, self.rows() as i32)));
        }
        if shape[1] == strides[0] as usize {
            return Ok(Layout::C((self.rows() as i32, self.cols() as i32)));
        }
        Err(StrideError::new(strides[0], strides[1]).into())
    }

    fn square_layout(&self) -> Result<Layout> {
        let l = self.layout()?;
        let (n, m) = l.size();
        if n == m {
            Ok(l)
        } else {
            Err(NotSquareError::new(n, m).into())
        }
    }

    fn as_allocated(&self) -> Result<&[A]> {
        Ok(self.as_slice_memory_order().ok_or(MemoryContError::new())?)
    }
}

impl<A, S> AllocatedArrayMut for ArrayBase<S, Ix2>
where
    S: DataMut<Elem = A>,
{
    fn as_allocated_mut(&mut self) -> Result<&mut [A]> {
        Ok(self.as_slice_memory_order_mut().ok_or(
            MemoryContError::new(),
        )?)
    }
}

pub fn into_col<S>(a: ArrayBase<S, Ix1>) -> ArrayBase<S, Ix2>
where
    S: Data,
{
    let n = a.len();
    a.into_shape((n, 1)).unwrap()
}

pub fn into_row<S>(a: ArrayBase<S, Ix1>) -> ArrayBase<S, Ix2>
where
    S: Data,
{
    let n = a.len();
    a.into_shape((1, n)).unwrap()
}

pub fn flatten<S>(a: ArrayBase<S, Ix2>) -> ArrayBase<S, Ix1>
where
    S: Data,
{
    let n = a.len();
    a.into_shape((n)).unwrap()
}

pub fn reconstruct<A, S>(l: Layout, a: Vec<A>) -> Result<ArrayBase<S, Ix2>>
where
    S: DataOwned<Elem = A>,
{
    Ok(ArrayBase::from_shape_vec(l.as_shape(), a)?)
}

pub fn uninitialized<A, S>(l: Layout) -> ArrayBase<S, Ix2>
where
    A: Copy,
    S: DataOwned<Elem = A>,
{
    unsafe { ArrayBase::uninitialized(l.as_shape()) }
}

pub fn replicate<A, Sv, So, D>(a: &ArrayBase<Sv, D>) -> ArrayBase<So, D>
where
    A: Copy,
    Sv: Data<Elem = A>,
    So: DataOwned<Elem = A> + DataMut,
    D: Dimension,
{
    let mut b = unsafe { ArrayBase::uninitialized(a.dim()) };
    b.assign(a);
    b
}

pub fn clone_with_layout<A, Si, So>(l: Layout, a: &ArrayBase<Si, Ix2>) -> ArrayBase<So, Ix2>
where
    A: Copy,
    Si: Data<Elem = A>,
    So: DataOwned<Elem = A> + DataMut,
{
    let mut b = uninitialized(l);
    b.assign(a);
    b
}

pub fn data_transpose<A, S>(a: &mut ArrayBase<S, Ix2>) -> Result<&mut ArrayBase<S, Ix2>>
where
    A: Copy,
    S: DataOwned<Elem = A> + DataMut,
{
    let l = a.layout()?.toggle_order();
    let new = clone_with_layout(l, a);
    ::std::mem::replace(a, new);
    Ok(a)
}
