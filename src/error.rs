//! Define Errors

use ndarray::{Ixs, ShapeError};

pub type Result<T> = ::std::result::Result<T, LinalgError>;

/// Master Error type of this crate
#[derive(Fail, Debug)]
pub enum LinalgError {
    /// Matrix is not square
    #[fail(display = "Not square: rows({}) != cols({})", rows, cols)]
    NotSquare { rows: i32, cols: i32 },

    /// LAPACK subroutine returns non-zero code
    #[fail(display = "LAPACK: return_code = {}", return_code)]
    LapackFailure { return_code: i32 },

    /// Strides of the array is not supported
    #[fail(display = "invalid stride: s0={}, s1={}", s0, s1)]
    InvalidStride { s0: Ixs, s1: Ixs },

    /// Memory is not aligned continously
    #[fail(display = "Memory is not contiguous")]
    MemoryNotCont {},

    /// Strides of the array is not supported
    #[fail(display = "Shape Error: {}", error)]
    ShapeFailure { error: ShapeError },
}

impl From<ShapeError> for LinalgError {
    fn from(error: ShapeError) -> LinalgError {
        LinalgError::ShapeFailure { error }
    }
}
