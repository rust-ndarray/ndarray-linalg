//! Singular-value decomposition (SVD)
//!
//! [Wikipedia article on SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition)

use ndarray::*;

use super::convert::*;
use super::error::*;
use super::layout::*;
use super::types::*;

#[derive(Clone, Copy, Eq, PartialEq)]
#[repr(u8)]
pub enum FlagSVD {
    All = b'A',
    // Overwrite = b'O',
    Some = b'S',
    None = b'N',
}

impl Into<FlagSVD> for bool {
    fn into(self) -> FlagSVD {
        if self {
            FlagSVD::All
        } else {
            FlagSVD::None
        }
    }
}

/// singular-value decomposition of matrix reference
pub trait SVD {
    type U;
    type VT;
    type Sigma;
    fn svd<X: Into<FlagSVD>, Y: Into<FlagSVD>>(
        &self,
        calc_u: X,
        calc_vt: Y,
    ) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)>;
    fn svd_dc<X: Into<FlagSVD>>(&self, mode: X) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)>;
}

/// singular-value decomposition
pub trait SVDInto {
    type U;
    type VT;
    type Sigma;
    fn svd_into<X: Into<FlagSVD>, Y: Into<FlagSVD>>(
        self,
        calc_u: X,
        calc_vt: Y,
    ) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)>;
    fn svd_dc_into<X: Into<FlagSVD>>(self, mode: X) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)>;
}

/// singular-value decomposition for mutable reference of matrix
pub trait SVDInplace {
    type U;
    type VT;
    type Sigma;
    fn svd_inplace<X: Into<FlagSVD>, Y: Into<FlagSVD>>(
        &mut self,
        calc_u: X,
        calc_vt: Y,
    ) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)>;
    fn svd_dc_inplace<X: Into<FlagSVD>>(&mut self, mode: X)
        -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)>;
}

impl<A, S> SVD for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
{
    type U = Array2<A>;
    type VT = Array2<A>;
    type Sigma = Array1<A::Real>;

    fn svd<X, Y>(&self, calc_u: X, calc_vt: Y) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)>
    where
        X: Into<FlagSVD>,
        Y: Into<FlagSVD>,
    {
        let a = self.to_owned();
        a.svd_into(calc_u, calc_vt)
    }

    fn svd_dc<X>(&self, mode: X) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)>
    where
        X: Into<FlagSVD>,
    {
        self.to_owned().svd_dc_into(mode)
    }
}

impl<A, S> SVDInto for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
{
    type U = Array2<A>;
    type VT = Array2<A>;
    type Sigma = Array1<A::Real>;

    fn svd_into<X, Y>(mut self, calc_u: X, calc_vt: Y) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)>
    where
        X: Into<FlagSVD>,
        Y: Into<FlagSVD>,
    {
        self.svd_inplace(calc_u, calc_vt)
    }

    fn svd_dc_into<X>(mut self, mode: X) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)>
    where
        X: Into<FlagSVD>,
    {
        self.svd_dc_inplace(mode)
    }
}

impl<A, S> SVDInplace for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
{
    type U = Array2<A>;
    type VT = Array2<A>;
    type Sigma = Array1<A::Real>;

    fn svd_inplace<X, Y>(&mut self, calc_u: X, calc_vt: Y) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)>
    where
        X: Into<FlagSVD>,
        Y: Into<FlagSVD>,
    {
        let l = self.layout()?;
        let svd_res = unsafe { A::svd(l, calc_u.into(), calc_vt.into(), self.as_allocated_mut()?)? };
        let (n, m) = l.size();
        let u = svd_res
            .u
            .map(|u| into_matrix(l.resized(n, n), u).expect("Size of U mismatches"));
        let vt = svd_res
            .vt
            .map(|vt| into_matrix(l.resized(m, m), vt).expect("Size of VT mismatches"));
        let s = ArrayBase::from_vec(svd_res.s);
        Ok((u, s, vt))
    }

    fn svd_dc_inplace<X>(&mut self, mode: X) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)>
    where
        X: Into<FlagSVD>,
    {
        let mode = mode.into();
        let l = self.layout()?;
        let svd_res = unsafe { A::svd_dc(l, mode, self.as_allocated_mut()?)? };
        let (m, n) = l.size();
        let k = m.min(n);
        let (ldu, tdu, ldvt, tdvt) = match mode {
            FlagSVD::All => (m, m, n, n),
            FlagSVD::Some => (m, k, k, n),
            FlagSVD::None => (1, 1, 1, 1),
        };
        let u = svd_res
            .u
            .map(|u| into_matrix(l.resized(ldu, tdu), u).expect("Size of U mismatches"));
        let vt = svd_res
            .vt
            .map(|vt| into_matrix(l.resized(ldvt, tdvt), vt).expect("Size of VT mismatches"));
        let s = ArrayBase::from_vec(svd_res.s);
        Ok((u, s, vt))
    }
}
