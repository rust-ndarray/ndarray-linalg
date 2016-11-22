//! Implement QR decomposition

use std::cmp::min;
use lapack::c::*;
use num_traits::Zero;

use error::LapackError;

pub trait ImplQR: Sized {
    fn qr(layout: Layout, n: usize, m: usize, mut a: Vec<Self>) -> Result<(Vec<Self>, Vec<Self>), LapackError>;
}

macro_rules! impl_qr {
    ($scalar:ty, $geqrf:path, $orgqr:path) => {
impl ImplQR for $scalar {
    fn qr(layout: Layout, n: usize, m: usize, mut a: Vec<Self>) -> Result<(Vec<Self>, Vec<Self>), LapackError> {
        let n = n as i32;
        let m = m as i32;
        let k = min(m, n);
        let lda = match layout {
            Layout::ColumnMajor => m,
            Layout::RowMajor => n,
        };
        let mut tau = vec![Self::zero(); k as usize];
        let info = $geqrf(layout, m, n, &mut a, lda, &mut tau);
        if info != 0 {
            return Err(From::from(info));
        }
        let r = a.clone();
        let info = $orgqr(layout, m, k, k, &mut a, lda, &mut tau);
        if info == 0 {
            Ok((a, r))
        } else {
            Err(From::from(info))
        }
    }
}
}} // endmacro

impl_qr!(f64, dgeqrf, dorgqr);
impl_qr!(f32, sgeqrf, sorgqr);
