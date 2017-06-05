
use ndarray::*;

use super::error::*;
use super::layout::{Layout, AllocatedArray, AllocatedArrayMut};
use impl2::LapackScalar;

pub trait SVD<U, S, VT> {
    fn svd(self, calc_u: bool, calc_vt: bool) -> Result<(Option<U>, S, Option<VT>)>;
}

impl<A, S, Su, Svt, Ss> SVD<ArrayBase<Su, Ix2>, ArrayBase<Ss, Ix1>, ArrayBase<Svt, Ix2>> for ArrayBase<S, Ix2>
    where A: LapackScalar,
          S: DataMut<Elem = A>,
          Su: DataOwned<Elem = A>,
          Svt: DataOwned<Elem = A>,
          Ss: DataOwned<Elem = A::Real>
{
    fn svd(mut self,
           calc_u: bool,
           calc_vt: bool)
           -> Result<(Option<ArrayBase<Su, Ix2>>, ArrayBase<Ss, Ix1>, Option<ArrayBase<Svt, Ix2>>)> {
        let n = self.rows();
        let m = self.cols();
        let l = self.layout()?;
        let svd_res = A::svd(l, calc_u, calc_vt, self.as_allocated_mut()?)?;
        let (u, vt) = match l {
            Layout::C(_) => {
                (svd_res.u.map(|u| ArrayBase::from_shape_vec((n, n), u).unwrap()),
                 svd_res.vt.map(|vt| ArrayBase::from_shape_vec((m, m), vt).unwrap()))
            }
            Layout::F(_) => {
                (svd_res.u.map(|u| ArrayBase::from_shape_vec((n, n).f(), u).unwrap()),
                 svd_res.vt.map(|vt| ArrayBase::from_shape_vec((m, m).f(), vt).unwrap()))
            }
        };
        let s = ArrayBase::from_vec(svd_res.s);
        Ok((u, s, vt))
    }
}
