//! Singular-value decomposition

use crate::{error::*, layout::MatrixLayout};
use cauchy::*;
use num_traits::Zero;

#[repr(u8)]
enum FlagSVD {
    All = b'A',
    // OverWrite = b'O',
    // Separately = b'S',
    No = b'N',
}

/// Result of SVD
pub struct SVDOutput<A: Scalar> {
    /// diagonal values
    pub s: Vec<A::Real>,
    /// Unitary matrix for destination space
    pub u: Option<Vec<A>>,
    /// Unitary matrix for departure space
    pub vt: Option<Vec<A>>,
}

/// Wraps `*gesvd`
pub trait SVD_: Scalar {
    unsafe fn svd(
        l: MatrixLayout,
        calc_u: bool,
        calc_vt: bool,
        a: &mut [Self],
    ) -> Result<SVDOutput<Self>>;
}

macro_rules! impl_svd {
    ($scalar:ty, $gesvd:path) => {
        impl SVD_ for $scalar {
            unsafe fn svd(
                l: MatrixLayout,
                calc_u: bool,
                calc_vt: bool,
                mut a: &mut [Self],
            ) -> Result<SVDOutput<Self>> {
                let (m, n) = l.size();
                let k = ::std::cmp::min(n, m);
                let lda = l.lda();
                let (ju, ldu, mut u) = if calc_u {
                    (FlagSVD::All, m, vec![Self::zero(); (m * m) as usize])
                } else {
                    (FlagSVD::No, 1, Vec::new())
                };
                let (jvt, ldvt, mut vt) = if calc_vt {
                    (FlagSVD::All, n, vec![Self::zero(); (n * n) as usize])
                } else {
                    (FlagSVD::No, n, Vec::new())
                };
                let mut s = vec![Self::Real::zero(); k as usize];
                let mut superb = vec![Self::Real::zero(); (k - 1) as usize];
                $gesvd(
                    l.lapacke_layout(),
                    ju as u8,
                    jvt as u8,
                    m,
                    n,
                    &mut a,
                    lda,
                    &mut s,
                    &mut u,
                    ldu,
                    &mut vt,
                    ldvt,
                    &mut superb,
                )
                .as_lapack_result()?;
                Ok(SVDOutput {
                    s,
                    u: if calc_u { Some(u) } else { None },
                    vt: if calc_vt { Some(vt) } else { None },
                })
            }
        }
    };
} // impl_svd!

impl_svd!(f64, lapacke::dgesvd);
impl_svd!(f32, lapacke::sgesvd);
impl_svd!(c64, lapacke::zgesvd);
impl_svd!(c32, lapacke::cgesvd);
