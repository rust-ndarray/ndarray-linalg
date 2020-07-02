use super::*;
use crate::{error::*, layout::MatrixLayout};
use cauchy::*;
use num_traits::Zero;

/// Specifies how many of the columns of *U* and rows of *V*ᵀ are computed and returned.
///
/// For an input array of shape *m*×*n*, the following are computed:
#[derive(Clone, Copy, Eq, PartialEq)]
#[repr(u8)]
pub enum UVTFlag {
    /// All *m* columns of *U* and all *n* rows of *V*ᵀ.
    Full = b'A',
    /// The first min(*m*,*n*) columns of *U* and the first min(*m*,*n*) rows of *V*ᵀ.
    Some = b'S',
    /// No columns of *U* or rows of *V*ᵀ.
    None = b'N',
}

pub trait SVDDC_: Scalar {
    unsafe fn svddc(l: MatrixLayout, jobz: UVTFlag, a: &mut [Self]) -> Result<SVDOutput<Self>>;
}

macro_rules! impl_svdd {
    ($scalar:ty, $gesdd:path) => {
        impl SVDDC_ for $scalar {
            unsafe fn svddc(
                l: MatrixLayout,
                jobz: UVTFlag,
                mut a: &mut [Self],
            ) -> Result<SVDOutput<Self>> {
                let (m, n) = l.size();
                let k = m.min(n);
                let lda = l.lda();
                let (ucol, vtrow) = match jobz {
                    UVTFlag::Full => (m, n),
                    UVTFlag::Some => (k, k),
                    UVTFlag::None => (1, 1),
                };
                let mut s = vec![Self::Real::zero(); k.max(1) as usize];
                let mut u = vec![Self::zero(); (m * ucol).max(1) as usize];
                let ldu = l.resized(m, ucol).lda();
                let mut vt = vec![Self::zero(); (vtrow * n).max(1) as usize];
                let ldvt = l.resized(vtrow, n).lda();
                $gesdd(
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
                )
                .as_lapack_result()?;
                Ok(SVDOutput {
                    s,
                    u: if jobz == UVTFlag::None { None } else { Some(u) },
                    vt: if jobz == UVTFlag::None {
                        None
                    } else {
                        Some(vt)
                    },
                })
            }
        }
    };
}

impl_svdd!(f32, lapacke::sgesdd);
impl_svdd!(f64, lapacke::dgesdd);
impl_svdd!(c32, lapacke::cgesdd);
impl_svdd!(c64, lapacke::zgesdd);
