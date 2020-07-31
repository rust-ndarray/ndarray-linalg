//! Define Errors

use ndarray::{Ixs, ShapeError};
use thiserror::Error;

pub type Result<T> = ::std::result::Result<T, LinalgError>;

/// Master Error type of this crate
#[derive(Debug, Error)]
pub enum LinalgError {
    /// Matrix is not square
    #[error("Not square: rows({}) != cols({})", rows, cols)]
    NotSquare { rows: i32, cols: i32 },

    /// LAPACK subroutine returns non-zero code
    #[error(transparent)]
    Lapack(#[from] lax::error::Error),

    /// Strides of the array is not supported
    #[error("invalid stride: s0={}, s1={}", s0, s1)]
    InvalidStride { s0: Ixs, s1: Ixs },

    /// Memory is not aligned continously
    #[error("Memroy is not continously")]
    MemoryNotCont,

    /// Obj cannot be made from a (rows, cols) matrix
    #[error("{} cannot be made from a ({}, {}) matrix", obj, rows, cols)]
    NotStandardShape {
        obj: &'static str,
        rows: i32,
        cols: i32,
    },

    /// Strides of the array is not supported
    #[error(transparent)]
    Shape(#[from] ShapeError),
}
