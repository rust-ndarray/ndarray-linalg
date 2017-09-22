//! utilities for convert array

use ndarray::*;

use super::error::*;
use super::lapack_traits::UPLO;
use super::layout::*;
use super::types::Conjugate;

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

pub fn into_matrix<A, S>(l: MatrixLayout, a: Vec<A>) -> Result<ArrayBase<S, Ix2>>
where
    S: DataOwned<Elem = A>,
{
    Ok(ArrayBase::from_shape_vec(l.as_shape(), a)?)
}

fn uninitialized<A, S>(l: MatrixLayout) -> ArrayBase<S, Ix2>
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

fn clone_with_layout<A, Si, So>(l: MatrixLayout, a: &ArrayBase<Si, Ix2>) -> ArrayBase<So, Ix2>
where
    A: Copy,
    Si: Data<Elem = A>,
    So: DataOwned<Elem = A> + DataMut,
{
    let mut b = uninitialized(l);
    b.assign(a);
    b
}

pub fn transpose_data<A, S>(a: &mut ArrayBase<S, Ix2>) -> Result<&mut ArrayBase<S, Ix2>>
where
    A: Copy,
    S: DataOwned<Elem = A> + DataMut,
{
    let l = a.layout()?.toggle_order();
    let new = clone_with_layout(l, a);
    ::std::mem::replace(a, new);
    Ok(a)
}

pub fn generalize<A, S, D>(a: Array<A, D>) -> ArrayBase<S, D>
where
    S: DataOwned<Elem = A>,
    D: Dimension,
{
    // FIXME
    // https://github.com/bluss/rust-ndarray/issues/325
    let strides: Vec<isize> = a.strides().to_vec();
    let new = if a.is_standard_layout() {
        ArrayBase::from_shape_vec(a.dim(), a.into_raw_vec()).unwrap()
    } else {
        ArrayBase::from_shape_vec(a.dim().f(), a.into_raw_vec()).unwrap()
    };
    assert_eq!(
        new.strides(),
        strides.as_slice(),
        "Custom stride is not supported"
    );
    new
}

/// Fills in the remainder of a Hermitian matrix that's represented by only one
/// triangle.
///
/// LAPACK methods on Hermitian matrices usually read/write only one triangular
/// portion of the matrix. This function fills in the other half based on the
/// data in the triangular portion corresponding to `uplo`.
///
/// ***Panics*** if `a` is not square.
pub(crate) fn triangular_fill_hermitian<A, S>(a: &mut ArrayBase<S, Ix2>, uplo: UPLO)
where
    A: Conjugate,
    S: DataMut<Elem = A>,
{
    assert!(a.is_square());
    match uplo {
        UPLO::Upper => {
            for row in 0..a.rows() {
                for col in 0..row {
                    a[(row, col)] = a[(col, row)].conj();
                }
            }
        }
        UPLO::Lower => {
            for col in 0..a.cols() {
                for row in 0..col {
                    a[(row, col)] = a[(col, row)].conj();
                }
            }
        }
    }
}
