//! Singular-value decomposition (SVD)
//!
//! [arXiv article on randomized linear algebra algorithms (including SVD)](https://arxiv.org/pdf/0909.4061.pdf)

use ndarray::linalg::Dot;
use ndarray::*;
use rand::{
    distributions::{
        uniform::{SampleUniform, Uniform},
        Distribution,
    },
    SeedableRng,
};

use super::error::*;
use super::qr::QR;
use super::svd::{FlagSVD, SVDInto};
use super::types::*;

#[cfg(feature = "sprs")]
use ::{sprs::{CsMatBase, SpIndex}, std::ops::Deref};

/// trait to capture shape of matrix
pub trait ArrayLike<D: ndarray::Dimension> {
    type A;
    fn dim(&self) -> D::Pattern;
}

impl<A, D, S> ArrayLike<D> for ArrayBase<S, D>
where
    D: ndarray::Dimension,
    S: DataMut<Elem = A>,
{
    type A = A;
    fn dim(&self) -> D::Pattern {
        ArrayBase::dim(self)
    }
}

#[cfg(feature = "sprs")]
impl<N, I, IptrStorage, IndStorage, DataStorage> ArrayLike<Ix2> for CsMatBase<N, I, IptrStorage, IndStorage, DataStorage>
where
    I: SpIndex,
    IptrStorage: Deref<Target = [I]>,
    IndStorage: Deref<Target = [I]>,
    DataStorage: Deref<Target = [N]>,
{
    type A = N;
    fn dim(&self) -> <Ix2 as ndarray::Dimension>::Pattern {
        (self.rows(), self.cols())
    }
}

/// randomized truncated singular-value decomposition
pub trait SVDRand {
    type U;
    type VT;
    type Sigma;
    fn svd_rand(
        &self,
        k: usize,
        n_iter: Option<usize>,
        l: Option<usize>,
        seed: Option<u64>,
    ) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)>;
}

impl<A, T> SVDRand for T
where
    A: Scalar + Lapack + SampleUniform,
    T: ArrayLike<Ix2, A = A> + Dot<Array2<A>, Output = Array2<A>>,
    for<'a> ArrayView2<'a, A>: Dot<T, Output = Array2<A>>,
    Array2<A>: QR<Q = Array2<A>> + Dot<T, Output = Array2<A>> + Dot<Array2<A>, Output = Array2<A>>,
{
    type U = Array2<A>;
    type VT = Array2<A>;
    type Sigma = Array1<A::Real>;

    fn svd_rand(
        &self,
        k: usize,
        n_iter: Option<usize>,
        l: Option<usize>,
        seed: Option<u64>,
    ) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>)> {
        let n_iter = n_iter.unwrap_or(7);
        let l = l.unwrap_or(k + 2);
        let (m, n) = self.dim();

        if m < 2 || n < 2 {
            panic!("m or n are <2!")
        }
        if m.min(n) < k {
            panic!("min(m, n) is <k");
        }

        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed.unwrap_or(0));
        let unif: Uniform<A> = Uniform::new(-A::one(), A::one());

        // TODO(nlhepler): Additional cases to handle
        // - fall through to straight svd when l/k is within ~25% of m or n.

        if m >= n {
            let omega: Array2<T::A> = Array2::from_shape_fn((n, l), move |_| unif.sample(&mut rng));
            let mut q = self.dot(&omega).qr()?.0;

            for _ in 0..n_iter {
                q = q.t().dot(self).reversed_axes().qr()?.0;
                q = self.dot(&q).qr()?.0;
            }

            let (u, s, vt) = {
                let b = q.t().dot(self);
                // info!("performing svd");
                let svd = b.svd_dc_into(FlagSVD::Some)?;
                // info!("svd finished");
                (
                    svd.0.unwrap().slice(s![.., ..k]).to_owned(),
                    svd.1.slice(s![..k]).to_owned(),
                    svd.2.unwrap().slice(s![..k, ..]).to_owned(),
                )
            };

            let u = q.dot(&u);
            Ok((Some(u), s, Some(vt)))
        } else {
            // n > m
            let omega = Array2::from_shape_fn((l, m), move |_| unif.sample(&mut rng));
            let mut q = omega.dot(self).reversed_axes().qr()?.0;

            for _ in 0..n_iter {
                q = self.dot(&q).qr()?.0;
                q = q.t().dot(self).reversed_axes().qr()?.0;
            }

            let (u, s, vt) = {
                let b = self.dot(&q);
                let svd = b.svd_dc_into(FlagSVD::Some)?;
                (
                    svd.0.unwrap().slice(s![.., ..k]).to_owned(),
                    svd.1.slice(s![..k]).to_owned(),
                    svd.2.unwrap().slice(s![..k, ..]).to_owned(),
                )
            };

            let vt = vt.dot(&q.t());
            Ok((Some(u), s, Some(vt)))
        }
    }
}
