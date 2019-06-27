//! Singular-value decomposition

use lapacke;
use num_traits::Zero;

use crate::error::*;
use crate::layout::MatrixLayout;
use crate::types::*;

use super::into_result;

use crate::svd::FlagSVD;

/// Result of SVD
pub struct SVDOutput<A: Scalar> {
    /// diagonal values
    pub s: Vec<A::Real>,
    /// Unitary matrix for destination space
    pub u: Option<Vec<A>>,
    /// Unitary matrix for departure space
    pub vt: Option<Vec<A>>,
}

/// Wraps `*gesvd` and `*gesdd`
pub trait SVD_: Scalar {
    unsafe fn svd(l: MatrixLayout, jobu: FlagSVD, jobvt: FlagSVD, a: &mut [Self]) -> Result<SVDOutput<Self>>;
    unsafe fn svd_dc(l: MatrixLayout, jobz: FlagSVD, a: &mut [Self]) -> Result<SVDOutput<Self>>;
}

macro_rules! impl_svd {
    ($scalar:ty, $gesvd:path, $gesdd:path) => {
        impl SVD_ for $scalar {
            unsafe fn svd(
                l: MatrixLayout,
                jobu: FlagSVD,
                jobvt: FlagSVD,
                mut a: &mut [Self],
            ) -> Result<SVDOutput<Self>> {
                let (m, n) = l.size();
                let k = ::std::cmp::min(n, m);
                let lda = l.lda();
                let ucol = match jobu {
                    FlagSVD::All => m,
                    FlagSVD::Some => k,
                    FlagSVD::None => 0,
                };
                let vtrow = match jobvt {
                    FlagSVD::All => n,
                    FlagSVD::Some => k,
                    FlagSVD::None => 0,
                };
                let mut u = vec![Self::zero(); (m * ucol).max(1) as usize];
                let ldu = l.resized(m, ucol).lda();
                let mut vt = vec![Self::zero(); (vtrow * n).max(1) as usize];
                let ldvt = l.resized(vtrow, n).lda();
                let mut s = vec![Self::Real::zero(); k as usize];
                let mut superb = vec![Self::Real::zero(); (k - 1) as usize];
                dbg!(ldvt);
                let info = $gesvd(
                    l.lapacke_layout(),
                    jobu as u8,
                    jobvt as u8,
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
                );
                into_result(
                    info,
                    SVDOutput {
                        s: s,
                        u: if jobu == FlagSVD::None { None } else { Some(u) },
                        vt: if jobvt == FlagSVD::None { None } else { Some(vt) },
                    },
                )
            }

            unsafe fn svd_dc(l: MatrixLayout, jobz: FlagSVD, mut a: &mut [Self]) -> Result<SVDOutput<Self>> {
                let (m, n) = l.size();
                let k = m.min(n);
                let lda = l.lda();
                let (ucol, vtrow) = match jobz {
                    FlagSVD::All => (m, n),
                    FlagSVD::Some => (k, k),
                    FlagSVD::None => (0, 0),
                };
                let mut s = vec![Self::Real::zero(); k.max(1) as usize];
                let mut u = vec![Self::zero(); (m * ucol).max(1) as usize];
                let ldu = l.resized(m, ucol).lda();
                let mut vt = vec![Self::zero(); (vtrow * n).max(1) as usize];
                let ldvt = l.resized(vtrow, n).lda();
                let info = $gesdd(
                    l.lapacke_layout(),
                    jobz as u8,
                    m,
                    n,
                    &mut a,
                    lda,
                    &mut s,
                    &mut u,
                    ldu,
                    &mut vt,
                    ldvt,
                );
                into_result(
                    info,
                    SVDOutput {
                        s: s,
                        u: if jobz == FlagSVD::None { None } else { Some(u) },
                        vt: if jobz == FlagSVD::None { None } else { Some(vt) },
                    },
                )
            }
        }
    };
} // impl_svd!

impl_svd!(f64, lapacke::dgesvd, lapacke::dgesdd);
impl_svd!(f32, lapacke::sgesvd, lapacke::sgesdd);
impl_svd!(c64, lapacke::zgesvd, lapacke::zgesdd);
impl_svd!(c32, lapacke::cgesvd, lapacke::cgesdd);
