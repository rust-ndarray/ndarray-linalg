use super::*;
use crate::{error::*, layout::MatrixLayout};
use cauchy::*;
use num_traits::{ToPrimitive, Zero};

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

macro_rules! impl_svddc_real {
    ($scalar:ty, $gesdd:path) => {
        impl SVDDC_ for $scalar {
            unsafe fn svddc(
                l: MatrixLayout,
                jobz: UVTFlag,
                mut a: &mut [Self],
            ) -> Result<SVDOutput<Self>> {
                let m = l.lda();
                let n = l.len();
                let k = m.min(n);
                let mut s = vec![Self::Real::zero(); k as usize];

                let (u_col, vt_row) = match jobz {
                    UVTFlag::Full | UVTFlag::None => (m, n),
                    UVTFlag::Some => (k, k),
                };
                let (mut u, mut vt) = match jobz {
                    UVTFlag::Full => (
                        Some(vec![Self::zero(); (m * m) as usize]),
                        Some(vec![Self::zero(); (n * n) as usize]),
                    ),
                    UVTFlag::Some => (
                        Some(vec![Self::zero(); (m * u_col) as usize]),
                        Some(vec![Self::zero(); (n * vt_row) as usize]),
                    ),
                    UVTFlag::None => (None, None),
                };

                // eval work size
                let mut info = 0;
                let mut iwork = vec![0; 8 * k as usize];
                let mut work_size = [Self::zero()];
                $gesdd(
                    jobz as u8,
                    m,
                    n,
                    &mut a,
                    m,
                    &mut s,
                    u.as_mut().map(|x| x.as_mut_slice()).unwrap_or(&mut []),
                    m,
                    vt.as_mut().map(|x| x.as_mut_slice()).unwrap_or(&mut []),
                    vt_row,
                    &mut work_size,
                    -1,
                    &mut iwork,
                    &mut info,
                );
                info.as_lapack_result()?;

                // do svd
                let lwork = work_size[0].to_usize().unwrap();
                let mut work = vec![Self::zero(); lwork];
                $gesdd(
                    jobz as u8,
                    m,
                    n,
                    &mut a,
                    m,
                    &mut s,
                    u.as_mut().map(|x| x.as_mut_slice()).unwrap_or(&mut []),
                    m,
                    vt.as_mut().map(|x| x.as_mut_slice()).unwrap_or(&mut []),
                    vt_row,
                    &mut work,
                    lwork as i32,
                    &mut iwork,
                    &mut info,
                );
                info.as_lapack_result()?;

                match l {
                    MatrixLayout::F { .. } => Ok(SVDOutput { s, u, vt }),
                    MatrixLayout::C { .. } => Ok(SVDOutput { s, u: vt, vt: u }),
                }
            }
        }
    };
}

impl_svddc_real!(f32, lapack::sgesdd);
impl_svddc_real!(f64, lapack::dgesdd);

macro_rules! impl_svddc_complex {
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

impl_svddc_complex!(c32, lapacke::cgesdd);
impl_svddc_complex!(c64, lapacke::zgesdd);
