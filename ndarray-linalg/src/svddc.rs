//! Singular-value decomposition (SVD) by divide-and-conquer (?gesdd)

use ndarray::*;

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
        let (n, m) = l.size();
        let k = std::cmp::min(n, m);
        let n = n as usize;
        let m = m as usize;
        let k = k as usize;

        let (u_col, vt_row) = match uvt_flag {
            UVTFlag::Full => (n, m),
            UVTFlag::Some => (k, k),
            UVTFlag::None => (0, 0),
        };

        let u = svd_res.u.map(|u| {
            assert_eq!(u.len(), n * u_col);
            match l {
                MatrixLayout::F { .. } => Array::from_shape_vec((n, u_col).f(), u),
                MatrixLayout::C { .. } => Array::from_shape_vec((n, u_col), u),
            }
            .unwrap()
        });

        let vt = svd_res.vt.map(|vt| {
            assert_eq!(vt.len(), m * vt_row);
            match l {
                MatrixLayout::F { .. } => Array::from_shape_vec((vt_row, m).f(), vt),
                MatrixLayout::C { .. } => Array::from_shape_vec((vt_row, m), vt),
            }
            .unwrap()
        });

        let s = ArrayBase::from(svd_res.s);
        Ok((u, s, vt))
    }
}
