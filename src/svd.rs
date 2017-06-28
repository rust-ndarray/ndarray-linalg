//! singular-value decomposition

use ndarray::*;

use super::convert::*;
use super::error::*;
use super::lapack_traits::LapackScalar;
use super::layout::*;

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
        (&mut self).svd(calc_u, calc_vt)
    }
}

impl<'a, A, S, Su, Svt, Ss> SVD<ArrayBase<Su, Ix2>, ArrayBase<Ss, Ix1>, ArrayBase<Svt, Ix2>> for &'a ArrayBase<S, Ix2>
    where A: LapackScalar + Clone,
          S: Data<Elem = A>,
          Su: DataOwned<Elem = A>,
          Svt: DataOwned<Elem = A>,
          Ss: DataOwned<Elem = A::Real>
{
    fn svd(self,
           calc_u: bool,
           calc_vt: bool)
           -> Result<(Option<ArrayBase<Su, Ix2>>, ArrayBase<Ss, Ix1>, Option<ArrayBase<Svt, Ix2>>)> {
        let a = self.to_owned();
        a.svd(calc_u, calc_vt)
    }
}

impl<'a, A, S, Su, Svt, Ss> SVD<ArrayBase<Su, Ix2>, ArrayBase<Ss, Ix1>, ArrayBase<Svt, Ix2>>
    for &'a mut ArrayBase<S, Ix2>
where
    A: LapackScalar,
    S: DataMut<Elem = A>,
    Su: DataOwned<Elem = A>,
    Svt: DataOwned<Elem = A>,
    Ss: DataOwned<Elem = A::Real>,
{
    fn svd(
        mut self,
        calc_u: bool,
        calc_vt: bool,
    ) -> Result<(Option<ArrayBase<Su, Ix2>>, ArrayBase<Ss, Ix1>, Option<ArrayBase<Svt, Ix2>>)> {
        let l = self.layout()?;
        let svd_res = A::svd(l, calc_u, calc_vt, self.as_allocated_mut()?)?;
        let (n, m) = l.size();
        let u = svd_res.u.map(|u| into_matrix(l.resized(n, n), u).unwrap());
        let vt = svd_res.vt.map(
            |vt| into_matrix(l.resized(m, m), vt).unwrap(),
        );
        let s = ArrayBase::from_vec(svd_res.s);
        Ok((u, s, vt))
    }
}
