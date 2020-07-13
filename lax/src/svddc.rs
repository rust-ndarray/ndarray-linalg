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

macro_rules! impl_svddc {
    (@real, $scalar:ty, $gesdd:path) => {
        impl_svddc!(@body, $scalar, $gesdd, );
    };
    (@complex, $scalar:ty, $gesdd:path) => {
        impl_svddc!(@body, $scalar, $gesdd, rwork);
    };
    (@body, $scalar:ty, $gesdd:path, $($rwork_ident:ident),*) => {
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

                $( // for complex only
                let mx = n.max(m) as usize;
                let mn = n.min(m) as usize;
                let lrwork = match jobz {
                    UVTFlag::None => 7 * mn,
                    _ => std::cmp::max(5*mn*mn + 5*mn, 2*mx*mn + 2*mn*mn + mn),
                };
                let mut $rwork_ident = vec![Self::Real::zero(); lrwork];
                )*

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
                    $(&mut $rwork_ident,)*
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
                    $(&mut $rwork_ident,)*
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

impl_svddc!(@real, f32, lapack::sgesdd);
impl_svddc!(@real, f64, lapack::dgesdd);
impl_svddc!(@complex, c32, lapack::cgesdd);
impl_svddc!(@complex, c64, lapack::zgesdd);
