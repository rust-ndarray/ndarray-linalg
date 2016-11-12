//! Implement eigenvalue decomposition of Hermite matrix

use lapack::fortran::*;
use num_traits::Zero;

use error::LapackError;

pub trait ImplEigh: Sized {
    fn eigh(n: usize, mut a: Vec<Self>) -> Result<(Vec<Self>, Vec<Self>), LapackError>;
}

macro_rules! impl_eigh {
    ($scalar:ty, $syev:path) => {
impl ImplEigh for $scalar {
    fn eigh(n: usize, mut a: Vec<Self>) -> Result<(Vec<Self>, Vec<Self>), LapackError> {
        let mut w = vec![Self::zero(); n];
        let mut work = vec![Self::zero(); 4 * n];
        let mut info = 0;
        $syev(b'V',
              b'U',
              n as i32,
              &mut a,
              n as i32,
              &mut w,
              &mut work,
              4 * n as i32,
              &mut info);
        if info == 0 {
            Ok((w, a))
        } else {
            Err(From::from(info))
        }
    }
}
}} // end macro_rules

impl_eigh!(f64, dsyev);
impl_eigh!(f32, ssyev);
