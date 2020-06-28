use num_traits::Zero;

use crate::error::*;
use crate::layout::MatrixLayout;
use crate::svddc::UVTFlag;
use crate::types::*;

use super::{into_result, SVDOutput};

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
                        s,
                        u: if jobz == UVTFlag::None { None } else { Some(u) },
                        vt: if jobz == UVTFlag::None {
                            None
                        } else {
                            Some(vt)
                        },
                    },
                )
            }
        }
    };
}

impl_svdd!(f32, lapacke::sgesdd);
impl_svdd!(f64, lapacke::dgesdd);
impl_svdd!(c32, lapacke::cgesdd);
impl_svdd!(c64, lapacke::zgesdd);
