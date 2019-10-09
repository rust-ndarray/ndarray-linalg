//! Singular-value decomposition (SVD)
//!
//! [Wikipedia article on SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition)

use ndarray::*;

use super::convert::*;
use super::error::*;
use super::layout::*;
use super::types::*;

/// singular-value decomposition of matrix reference
pub trait SVD {
    type U;
    type VT;
    type Sigma;
    fn svd(&self, calc_u: bool, calc_vt: bool) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)>;
}

/// singular-value decomposition
pub trait SVDInto {
    type U;
    type VT;
    type Sigma;
    fn svd_into(self, calc_u: bool, calc_vt: bool) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)>;
}

/// singular-value decomposition for mutable reference of matrix
pub trait SVDInplace {
    type U;
    type VT;
    type Sigma;
    fn svd_inplace(&mut self, calc_u: bool, calc_vt: bool) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)>;
}

impl<A, S> SVDInto for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
{
    type U = Array2<A>;
    type VT = Array2<A>;
    type Sigma = Array1<A::Real>;

    fn svd_into(mut self, calc_u: bool, calc_vt: bool) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)> {
        self.svd_inplace(calc_u, calc_vt)
    }
}

impl<A, S> SVD for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
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

impl<A, S> SVDInplace for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
{
    type U = Array2<A>;
    type VT = Array2<A>;
    type Sigma = Array1<A::Real>;

    fn svd_inplace(&mut self, calc_u: bool, calc_vt: bool) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)> {
        let l = self.layout()?;
        let svd_res = unsafe { A::svd(l, calc_u, calc_vt, self.as_allocated_mut()?)? };
        let (n, m) = l.size();
        let u = svd_res
            .u
            .map(|u| into_matrix(l.resized(n, n), u).expect("Size of U mismatches"));
        let vt = svd_res
            .vt
            .map(|vt| into_matrix(l.resized(m, m), vt).expect("Size of VT mismatches"));
        let s = ArrayBase::from(svd_res.s);
        Ok((u, s, vt))
    }
}
