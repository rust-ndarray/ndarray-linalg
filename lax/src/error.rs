use thiserror::Error;

pub type Result<T> = ::std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error(
        "Invalid value for LAPACK subroutine {}-th argument",
        -return_code
    )]
    LapackInvalidValue { return_code: i32 },

    #[error(
        "Comutational failure in LAPACK subroutine: return_code = {}",
        return_code
    )]
    LapackComputationalFailure { return_code: i32 },

    /// Strides of the array is not supported
    #[error("Invalid shape")]
    InvalidShape,
}

pub trait AsLapackResult {
    fn as_lapack_result(self) -> Result<()>;
}

impl AsLapackResult for i32 {
    fn as_lapack_result(self) -> Result<()> {
        if self > 0 {
            return Err(Error::LapackComputationalFailure { return_code: self });
        }
        if self < 0 {
            return Err(Error::LapackInvalidValue { return_code: self });
        }
        Ok(())
    }
}
