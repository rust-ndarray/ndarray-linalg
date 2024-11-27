//! utilities for convert array

use lax::UPLO;
use ndarray::*;

use super::error::*;
use super::layout::*;
use super::types::*;

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
    a.into_shape(n).unwrap()
}

pub fn into_matrix<A, S>(l: MatrixLayout, a: Vec<A>) -> Result<ArrayBase<S, Ix2>>
where
    S: DataOwned<Elem = A>,
{
    match l {
        MatrixLayout::C { row, lda } => {
            Ok(ArrayBase::from_shape_vec((row as usize, lda as usize), a)?)
        }
        MatrixLayout::F { col, lda } => Ok(ArrayBase::from_shape_vec(
            (lda as usize, col as usize).f(),
            a,
        )?),
    }
}

pub fn replicate<A, Sv, So, D>(a: &ArrayBase<Sv, D>) -> ArrayBase<So, D>
where
    A: Copy,
    Sv: Data<Elem = A>,
    So: DataOwned<Elem = A> + DataMut,
    D: Dimension,
{
    unsafe {
        let ret = ArrayBase::<So, D>::build_uninit(a.dim(), |view| {
            a.assign_to(view);
        });
        ret.assume_init()
    }
}

fn clone_with_layout<A, Si, So>(l: MatrixLayout, a: &ArrayBase<Si, Ix2>) -> ArrayBase<So, Ix2>
where
    A: Copy,
    Si: Data<Elem = A>,
    So: DataOwned<Elem = A> + DataMut,
{
    let shape_builder = match l {
        MatrixLayout::C { row, lda } => (row as usize, lda as usize).set_f(false),
        MatrixLayout::F { col, lda } => (lda as usize, col as usize).set_f(true),
    };
    unsafe {
        let ret = ArrayBase::<So, _>::build_uninit(shape_builder, |view| {
            a.assign_to(view);
        });
        ret.assume_init()
    }
}

pub fn transpose_data<A, S>(a: &mut ArrayBase<S, Ix2>) -> Result<&mut ArrayBase<S, Ix2>>
where
    A: Copy,
    S: DataOwned<Elem = A> + DataMut,
{
    let l = a.layout()?.toggle_order();
    let new = clone_with_layout(l, a);
    *a = new;
    Ok(a)
}

pub fn generalize<A, S, D>(a: Array<A, D>) -> ArrayBase<S, D>
where
    S: DataOwned<Elem = A>,
    D: Dimension,
{
    // FIXME
    // https://github.com/bluss/rust-ndarray/issues/325
    //
    // copy strides
    let mut strides = D::zeros(a.ndim());
    for (index, &s) in a.strides().iter().enumerate() {
        strides[index] = s as usize;
    }
    let a_dim = a.raw_dim();
    let a_len = a.len();
    let data = a.into_raw_vec();
    assert_eq!(
        a_len,
        data.len(),
        "generalize: non-contig arrays are not supported"
    );
    ArrayBase::from_shape_vec(a_dim.strides(strides), data).unwrap()
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
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
{
    assert!(a.is_square());
    match uplo {
        UPLO::Upper => {
            for row in 0..a.nrows() {
                for col in 0..row {
                    a[(row, col)] = a[(col, row)].conj();
                }
            }
        }
        UPLO::Lower => {
            for col in 0..a.ncols() {
                for row in 0..col {
                    a[(row, col)] = a[(col, row)].conj();
                }
            }
        }
    }
}
