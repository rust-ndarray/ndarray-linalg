//! implement Cholesky decomposition

use lapack::c;

use types::*;
use error::*;
use layout::Layout;

use super::{into_result, UPLO};

pub trait Cholesky_: Sized {
    fn cholesky(Layout, UPLO, a: &mut [Self]) -> Result<()>;
}

macro_rules! impl_cholesky {
    ($scalar:ty, $potrf:path) => {
impl Cholesky_ for $scalar {
    fn cholesky(l: Layout, uplo: UPLO, mut a: &mut [Self]) -> Result<()> {
        let (n, _) = l.size();
        let info = $potrf(l.lapacke_layout(), uplo as u8, n, &mut a, n);
        into_result(info, ())
    }
}
}} // end macro_rules

impl_cholesky!(f64, c::dpotrf);
impl_cholesky!(f32, c::spotrf);
impl_cholesky!(c64, c::zpotrf);
impl_cholesky!(c32, c::cpotrf);
