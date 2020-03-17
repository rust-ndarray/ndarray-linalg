///! Implements truncated singular value decomposition
///

use std::ops::DivAssign;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use num_traits::{Float, NumCast};
use crate::{Scalar, Lapack};
use super::lobpcg::{lobpcg, EigResult, Order};
use crate::error::Result;

#[derive(Debug)]
pub struct TruncatedSvdResult<A> {
    eigvals: Array1<A>,
    eigvecs: Array2<A>,
    problem: Array2<A>,
    ngm: bool
}

impl<A: Float + PartialOrd + DivAssign<A> + 'static> TruncatedSvdResult<A> {
    fn singular_values_with_indices(&self) -> (Array1<A>, Vec<usize>) {
        let mut a = self.eigvals.iter()
            .map(|x| x.sqrt())
            .enumerate()
            .collect::<Vec<_>>();

        a.sort_by(|(_,x), (_, y)| x.partial_cmp(&y).unwrap().reverse());
        
        let (values, indices): (Vec<A>, Vec<usize>) = a.into_iter()
            .filter(|(_,x)| *x > NumCast::from(1e-5).unwrap())
            .map(|(a,b)| (b,a))
            .unzip();

        (Array1::from(values), indices)
    }

    pub fn values(&self) -> Array1<A> {
        let (values, _) = self.singular_values_with_indices();

        values
    }

    pub fn values_vecs(&self) -> (Array2<A>, Array1<A>, Array2<A>) {
        let (values, indices) = self.singular_values_with_indices();

        let (u, v) = if self.ngm {
            let vlarge = self.eigvecs.select(Axis(1), &indices);
            let mut ularge = self.problem.dot(&vlarge);
            
            ularge.gencolumns_mut().into_iter()
                .zip(values.iter()) 
                .for_each(|(mut a,b)| a.mapv_inplace(|x| x / *b));

            (ularge, vlarge)
        } else {
            let ularge = self.eigvecs.select(Axis(1), &indices);

            let mut vlarge = self.problem.t().dot(&ularge);
            vlarge.gencolumns_mut().into_iter()
                .zip(values.iter()) 
                .for_each(|(mut a,b)| a.mapv_inplace(|x| x / *b));

            (ularge, vlarge)
        };

        (u, values, v.reversed_axes())
    }
}

/// Truncated singular value decomposition
///
/// This struct wraps the LOBPCG algorithm and provides convenient builder-pattern access to
/// parameter like maximal iteration, precision and constraint matrix. Furthermore it allows
/// conversion into a iterative solver where each iteration step yields a new eigenvalue/vector
/// pair.
pub struct TruncatedSvd<A: Scalar> {
    order: Order,
    problem: Array2<A>,
    precision: A::Real,
    maxiter: usize
}

impl<A: Scalar + Lapack + PartialOrd + Default> TruncatedSvd<A> {
    pub fn new(problem: Array2<A>, order: Order) -> TruncatedSvd<A> {
        TruncatedSvd {
            precision: NumCast::from(1e-5).unwrap(),
            maxiter: problem.len_of(Axis(0)) * 2,
            order, 
            problem
        }
    }

    pub fn precision(mut self, precision: A::Real) -> Self {
        self.precision = precision;

        self
    }

    pub fn maxiter(mut self, maxiter: usize) -> Self {
        self.maxiter = maxiter;

        self

    }

    // calculate the eigenvalues once
    pub fn once(&self, num: usize) -> Result<TruncatedSvdResult<A>> {
        let (n,m) = (self.problem.nrows(), self.problem.ncols());

        let x = Array2::random((usize::min(n,m), num), Uniform::new(0.0, 1.0))
            .mapv(|x| NumCast::from(x).unwrap());

        let res = if n > m {
            lobpcg(|y| self.problem.t().dot(&self.problem.dot(&y)), x, None, None, self.precision, self.maxiter, self.order.clone())
        } else {
            lobpcg(|y| self.problem.dot(&self.problem.t().dot(&y)), x, None, None, self.precision, self.maxiter, self.order.clone())
        };

        match res {
            EigResult::Ok(vals, vecs, _) | EigResult::Err(vals, vecs, _, _) => {
                Ok(TruncatedSvdResult {
                    problem: self.problem.clone(),
                    eigvals: vals,
                    eigvecs: vecs,
                    ngm: n > m
                })
            },
            EigResult::NoResult(err) => Err(err)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::close_l2;
    use super::TruncatedSvd;
    use super::Order;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_truncated_svd() {
        let a = arr2(&[[3., 2., 2.],
                       [2., 3., -2.]]);

        let res = TruncatedSvd::new(a, Order::Largest)
            .precision(1e-5)
            .maxiter(500)
            .once(2)
            .unwrap();
        
        let (u, sigma, v_t) = res.values_vecs();

        close_l2(&sigma, &arr1(&[5.0, 3.0]), 1e-5);
    }
}
