//! Singular-value decomposition (SVD)
//!
//! [Wikipedia article on SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition)

use ndarray::*;

use super::error::*;
use super::layout::*;
use super::types::*;

/// singular-value decomposition of matrix reference
pub trait SVD {
    type U;
    type VT;
    type Sigma;
    fn svd(
        &self,
        calc_u: bool,
        calc_vt: bool,
    ) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)>;
}

/// singular-value decomposition
pub trait SVDInto {
    type U;
    type VT;
    type Sigma;
    fn svd_into(
        self,
        calc_u: bool,
        calc_vt: bool,
    ) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)>;
}

/// singular-value decomposition for mutable reference of matrix
pub trait SVDInplace {
    type U;
    type VT;
    type Sigma;
    fn svd_inplace(
        &mut self,
        calc_u: bool,
        calc_vt: bool,
    ) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)>;
}

impl<A, S> SVDInto for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
{
    type U = Array2<A>;
    type VT = Array2<A>;
    type Sigma = Array1<A::Real>;

    fn svd_into(
        mut self,
        calc_u: bool,
        calc_vt: bool,
    ) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)> {
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

    fn svd(
        &self,
        calc_u: bool,
        calc_vt: bool,
    ) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)> {
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

    fn svd_inplace(
        &mut self,
        calc_u: bool,
        calc_vt: bool,
    ) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)> {
        let l = self.layout()?;
        let svd_res = unsafe { A::svd(l, calc_u, calc_vt, self.as_allocated_mut()?)? };
        let (n, m) = l.size();
        let n = n as usize;
        let m = m as usize;

        let u = svd_res.u.map(|u| {
            assert_eq!(u.len(), n * n);
            match l {
                MatrixLayout::F { .. } => Array::from_shape_vec((n, n).f(), u),
                MatrixLayout::C { .. } => Array::from_shape_vec((n, n), u),
            }
            .unwrap()
        });

        let vt = svd_res.vt.map(|vt| {
            assert_eq!(vt.len(), m * m);
            match l {
                MatrixLayout::F { .. } => Array::from_shape_vec((m, m).f(), vt),
                MatrixLayout::C { .. } => Array::from_shape_vec((m, m), vt),
            }
            .unwrap()
        });

        let s = ArrayBase::from(svd_res.s);
        Ok((u, s, vt))
    }
}
