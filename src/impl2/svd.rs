//! Implement Operator norms for matrices

use lapack::c;
use num_traits::Zero;

use types::*;
use error::*;
use layout::Layout;

use super::into_result;

#[repr(u8)]
enum FlagSVD {
    All = b'A',
    // OverWrite = b'O',
    // Separately = b'S',
    No = b'N',
}

pub struct SVDOutput<A: AssociatedReal> {
    pub s: Vec<A::Real>,
    pub u: Option<Vec<A>>,
    pub vt: Option<Vec<A>>,
}

pub trait SVD_: AssociatedReal {
    fn svd(Layout, calc_u: bool, calc_vt: bool, a: &mut [Self]) -> Result<SVDOutput<Self>>;
}

macro_rules! impl_svd {
    ($scalar:ty, $gesvd:path) => {

impl SVD_ for $scalar {
    fn svd(l: Layout, calc_u: bool, calc_vt: bool, mut a: &mut [Self]) -> Result<SVDOutput<Self>> {
        let (m, n) = l.size();
        let k = ::std::cmp::min(n, m);
        let lda = l.lda();
        let (ju, ldu, mut u) = if calc_u {
            (FlagSVD::All, m, vec![Self::zero(); (m*m) as usize])
        } else {
            (FlagSVD::No, 0, Vec::new())
        };
        let (jvt, ldvt, mut vt) = if calc_vt {
            (FlagSVD::All, n, vec![Self::zero(); (n*n) as usize])
        } else {
            (FlagSVD::No, 0, Vec::new())
        };
        let mut s = vec![Self::Real::zero(); k as usize];
        let mut superb = vec![Self::Real::zero(); (k-2) as usize];
        let info = $gesvd(l.lapacke_layout(), ju as u8, jvt as u8, m, n, &mut a, lda, &mut s, &mut u, ldu, &mut vt, ldvt, &mut superb);
        into_result(info, SVDOutput {
            s: s,
            u: if ldu > 0 { Some(u) } else { None },
            vt: if ldvt > 0 { Some(vt) } else { None },
        })
    }
}

}} // impl_svd!

impl_svd!(f64, c::dgesvd);
impl_svd!(f32, c::sgesvd);
impl_svd!(c64, c::zgesvd);
impl_svd!(c32, c::cgesvd);
