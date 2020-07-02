//! Singular-value decomposition (SVD) by divide-and-conquer (?gesdd)

use ndarray::*;

use super::convert::*;
use super::error::*;
use super::layout::*;
use super::types::*;

pub use lapack::svddc::UVTFlag;

/// Singular-value decomposition of matrix (copying) by divide-and-conquer
pub trait SVDDC {
    type U;
    type VT;
    type Sigma;
    fn svddc(&self, uvt_flag: UVTFlag) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)>;
}

/// Singular-value decomposition of matrix by divide-and-conquer
pub trait SVDDCInto {
    type U;
    type VT;
    type Sigma;
    fn svddc_into(
        self,
        uvt_flag: UVTFlag,
    ) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)>;
}

/// Singular-value decomposition of matrix reference by divide-and-conquer
pub trait SVDDCInplace {
    type U;
    type VT;
    type Sigma;
    fn svddc_inplace(
        &mut self,
        uvt_flag: UVTFlag,
    ) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)>;
}

impl<A, S> SVDDC for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
{
    type U = Array2<A>;
    type VT = Array2<A>;
    type Sigma = Array1<A::Real>;

    fn svddc(&self, uvt_flag: UVTFlag) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)> {
        self.to_owned().svddc_into(uvt_flag)
    }
}

impl<A, S> SVDDCInto for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
{
    type U = Array2<A>;
    type VT = Array2<A>;
    type Sigma = Array1<A::Real>;

    fn svddc_into(
        mut self,
        uvt_flag: UVTFlag,
    ) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)> {
        self.svddc_inplace(uvt_flag)
    }
}

impl<A, S> SVDDCInplace for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
{
    type U = Array2<A>;
    type VT = Array2<A>;
    type Sigma = Array1<A::Real>;

    fn svddc_inplace(
        &mut self,
        uvt_flag: UVTFlag,
    ) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)> {
        let l = self.layout()?;
        let svd_res = unsafe { A::svddc(l, uvt_flag, self.as_allocated_mut()?)? };
        let (m, n) = l.size();
        let k = m.min(n);
        let (ldu, tdu, ldvt, tdvt) = match uvt_flag {
            UVTFlag::Full => (m, m, n, n),
            UVTFlag::Some => (m, k, k, n),
            UVTFlag::None => (1, 1, 1, 1),
        };
        let u = svd_res
            .u
            .map(|u| into_matrix(l.resized(ldu, tdu), u).expect("Size of U mismatches"));
        let vt = svd_res
            .vt
            .map(|vt| into_matrix(l.resized(ldvt, tdvt), vt).expect("Size of VT mismatches"));
        let s = ArrayBase::from(svd_res.s);
        Ok((u, s, vt))
    }
}
