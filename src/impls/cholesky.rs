
use lapack::c::*;
use error::LapackError;

pub trait ImplCholesky: Sized {
    fn cholesky(layout: Layout, n: usize, a: Vec<Self>) -> Result<Vec<Self>, LapackError>;
}

macro_rules! impl_cholesky {
    ($scalar:ty, $potrf:path) => {
impl ImplCholesky for $scalar {
    fn cholesky(layout: Layout, n: usize, mut a: Vec<Self>) -> Result<Vec<Self>, LapackError> {
        let info = $potrf(layout, b'U', n as i32, &mut a, n as i32);
        if info == 0 {
            Ok(a)
        } else {
            Err(From::from(info))
        }
    }
}
}} // end macro_rules

impl_cholesky!(f64, dpotrf);
impl_cholesky!(f32, spotrf);
