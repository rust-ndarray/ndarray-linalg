//! singular-value decomposition

use ndarray::*;

use super::convert::*;
use super::error::*;
use super::layout::*;
use super::types::*;

pub trait SVD {
    type U;
    type VT;
    type Sigma;
    fn svd(&self, calc_u: bool, calc_vt: bool) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)>;
}

pub trait SVDInto {
    type U;
    type VT;
    type Sigma;
    fn svd_into(self, calc_u: bool, calc_vt: bool) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)>;
}

pub trait SVDMut {
    type U;
    type VT;
    type Sigma;
    fn svd_mut(&mut self, calc_u: bool, calc_vt: bool) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)>;
}

impl<A, S> SVDInto for ArrayBase<S, Ix2>
where
    A: Scalar,
    S: DataMut<Elem = A>,
{
    type U = Array2<A>;
    type VT = Array2<A>;
    type Sigma = Array1<A::Real>;

    fn svd_into(mut self, calc_u: bool, calc_vt: bool) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)> {
        self.svd_mut(calc_u, calc_vt)
    }
}

impl<A, S> SVD for ArrayBase<S, Ix2>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    type U = Array2<A>;
    type VT = Array2<A>;
    type Sigma = Array1<A::Real>;

    fn svd(&self, calc_u: bool, calc_vt: bool) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)> {
        let a = self.to_owned();
        a.svd_into(calc_u, calc_vt)
    }
}

impl<A, S> SVDMut for ArrayBase<S, Ix2>
where
    A: Scalar,
    S: DataMut<Elem = A>,
{
    type U = Array2<A>;
    type VT = Array2<A>;
    type Sigma = Array1<A::Real>;

    fn svd_mut(&mut self, calc_u: bool, calc_vt: bool) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)> {
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
