use ndarray::*;

use super::error::*;
use super::layout::*;

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
