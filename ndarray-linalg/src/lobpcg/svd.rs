///! Truncated singular value decomposition
///!
///! This module computes the k largest/smallest singular values/vectors for a dense matrix.
use super::lobpcg::{lobpcg, LobpcgResult, Order};
use crate::error::Result;
use crate::generate;
use cauchy::Scalar;
use lax::Lapack;
use ndarray::prelude::*;
use ndarray::ScalarOperand;
use num_traits::{Float, NumCast};
use std::ops::DivAssign;

/// The result of a eigenvalue decomposition, not yet transformed into singular values/vectors
///
/// Provides methods for either calculating just the singular values with reduced cost or the
/// vectors with additional cost of matrix multiplication.
#[derive(Debug)]
pub struct TruncatedSvdResult<A> {
    eigvals: Array1<A>,
    eigvecs: Array2<A>,
    problem: Array2<A>,
    ngm: bool,
}

impl<A: Float + PartialOrd + DivAssign<A> + 'static + MagnitudeCorrection> TruncatedSvdResult<A> {
    /// Returns singular values ordered by magnitude with indices
    fn singular_values_with_indices(&self) -> (Array1<A>, Vec<usize>) {
        // numerate eigenvalues
        let mut a = self.eigvals.iter().enumerate().collect::<Vec<_>>();

        // sort by magnitude
        a.sort_by(|(_, x), (_, y)| x.partial_cmp(&y).unwrap().reverse());

        // calculate cut-off magnitude (borrowed from scipy)
        let cutoff = A::epsilon() * // float precision
                     A::correction() * // correction term (see trait below)
                     *a[0].1; // max eigenvalue

        // filter low singular values away
        let (values, indices): (Vec<A>, Vec<usize>) = a
            .into_iter()
            .filter(|(_, x)| *x > &cutoff)
            .map(|(a, b)| (b.sqrt(), a))
            .unzip();

        (Array1::from(values), indices)
    }

    /// Returns singular values ordered by magnitude
    pub fn values(&self) -> Array1<A> {
        let (values, _) = self.singular_values_with_indices();

        values
    }

    /// Returns singular values, left-singular vectors and right-singular vectors
    pub fn values_vectors(&self) -> (Array2<A>, Array1<A>, Array2<A>) {
        let (values, indices) = self.singular_values_with_indices();

        // branch n > m (for A is [n x m])
        let (u, v) = if self.ngm {
            let vlarge = self.eigvecs.select(Axis(1), &indices);
            let mut ularge = self.problem.dot(&vlarge);

            ularge
                .columns_mut()
                .into_iter()
                .zip(values.iter())
                .for_each(|(mut a, b)| a.mapv_inplace(|x| x / *b));

            (ularge, vlarge)
        } else {
            let ularge = self.eigvecs.select(Axis(1), &indices);

            let mut vlarge = self.problem.t().dot(&ularge);
            vlarge
                .columns_mut()
                .into_iter()
                .zip(values.iter())
                .for_each(|(mut a, b)| a.mapv_inplace(|x| x / *b));

            (ularge, vlarge)
        };

        (u, values, v.reversed_axes())
    }
}

/// Truncated singular value decomposition
///
/// Wraps the LOBPCG algorithm and provides convenient builder-pattern access to
/// parameter like maximal iteration, precision and contrain matrix.
pub struct TruncatedSvd<A: Scalar> {
    order: Order,
    problem: Array2<A>,
    precision: f32,
    maxiter: usize,
}

impl<A: Float + Scalar + ScalarOperand + Lapack + PartialOrd + Default> TruncatedSvd<A> {
    /// Create a new truncated SVD problem
    ///
    /// # Parameters
    ///  * `problem`: rectangular matrix which is decomposed
    ///  * `order`: whether to return large or small (close to zero) singular values
    pub fn new(problem: Array2<A>, order: Order) -> TruncatedSvd<A> {
        TruncatedSvd {
            precision: 1e-5,
            maxiter: problem.len_of(Axis(0)) * 2,
            order,
            problem,
        }
    }

    /// Set the required precision of the solution
    ///
    /// The precision is, in the context of SVD, the square-root precision of the underlying
    /// eigenproblem solution. The eigenproblem-precision is used to check the L2 error of each
    /// eigenvector and stops its optimization when the required precision is reached.
    pub fn precision(mut self, precision: f32) -> Self {
        self.precision = precision;

        self
    }

    /// Set the maximal number of iterations
    ///
    /// The LOBPCG is an iterative approach to eigenproblems and stops when this maximum
    /// number of iterations are reached.
    pub fn maxiter(mut self, maxiter: usize) -> Self {
        self.maxiter = maxiter;

        self
    }

    /// Calculate the singular value decomposition
    ///
    /// # Parameters
    ///
    ///  * `num`: number of singular-value/vector pairs, ordered by magnitude
    ///
    /// # Example
    ///
    /// ```rust
    /// use ndarray::{arr1, Array2};
    /// use ndarray_linalg::{TruncatedSvd, TruncatedOrder};
    ///
    /// let diag = arr1(&[1., 2., 3., 4., 5.]);
    /// let a = Array2::from_diag(&diag);
    ///
    /// let eig = TruncatedSvd::new(a, TruncatedOrder::Largest)
    ///    .precision(1e-5)
    ///    .maxiter(500);
    ///
    /// let res = eig.decompose(3);
    /// ```
    pub fn decompose(self, num: usize) -> Result<TruncatedSvdResult<A>> {
        if num < 1 {
            panic!("The number of singular values to compute should be larger than zero!");
        }

        let (n, m) = (self.problem.nrows(), self.problem.ncols());

        // generate initial matrix
        let x: Array2<f32> = generate::random((usize::min(n, m), num));
        let x = x.mapv(|x| NumCast::from(x).unwrap());

        // square precision because the SVD squares the eigenvalue as well
        let precision = self.precision * self.precision;

        // use problem definition with less operations required
        let res = if n > m {
            lobpcg(
                |y| self.problem.t().dot(&self.problem.dot(&y)),
                x,
                |_| {},
                None,
                precision,
                self.maxiter,
                self.order.clone(),
            )
        } else {
            lobpcg(
                |y| self.problem.dot(&self.problem.t().dot(&y)),
                x,
                |_| {},
                None,
                precision,
                self.maxiter,
                self.order.clone(),
            )
        };

        // convert into TruncatedSvdResult
        match res {
            LobpcgResult::Ok(vals, vecs, _) | LobpcgResult::Err(vals, vecs, _, _) => {
                Ok(TruncatedSvdResult {
                    problem: self.problem,
                    eigvals: vals,
                    eigvecs: vecs,
                    ngm: n > m,
                })
            }
            LobpcgResult::NoResult(err) => Err(err),
        }
    }
}

/// Magnitude Correction
///
/// The magnitude correction changes the cut-off point at which an eigenvector belongs to the
/// null-space and its eigenvalue is therefore zero. The correction is multiplied by the floating
/// point epsilon and therefore dependent on the floating type.
pub trait MagnitudeCorrection {
    fn correction() -> Self;
}

impl MagnitudeCorrection for f32 {
    fn correction() -> Self {
        1.0e3
    }
}

impl MagnitudeCorrection for f64 {
    fn correction() -> Self {
        1.0e6
    }
}

#[cfg(test)]
mod tests {
    use super::Order;
    use super::TruncatedSvd;
    use crate::{close_l2, generate};

    use ndarray::{arr1, arr2, Array1, Array2};
    use ndarray_rand::{rand_distr::StandardNormal, RandomExt};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256Plus;

    use approx::assert_abs_diff_eq;

    #[test]
    fn test_truncated_svd() {
        let a = arr2(&[[3., 2., 2.], [2., 3., -2.]]);

        let res = TruncatedSvd::new(a, Order::Largest)
            .precision(1e-5)
            .maxiter(10)
            .decompose(2)
            .unwrap();

        let (_, sigma, _) = res.values_vectors();

        close_l2(&sigma, &arr1(&[5.0, 3.0]), 1e-5);
    }

    #[test]
    fn test_truncated_svd_random() {
        let a: Array2<f64> = generate::random((50, 10));

        let res = TruncatedSvd::new(a.clone(), Order::Largest)
            .precision(1e-5)
            .maxiter(10)
            .decompose(10)
            .unwrap();

        let (u, sigma, v_t) = res.values_vectors();
        let reconstructed = u.dot(&Array2::from_diag(&sigma).dot(&v_t));

        close_l2(&a, &reconstructed, 1e-5);
    }

    /// Eigenvalue structure in high dimensions
    ///
    /// This test checks that the eigenvalues are following the Marchensko-Pastur law. The data is
    /// standard uniformly distributed (i.e. E(x) = 0, E^2(x) = 1) and we have twice the amount of
    /// data when compared to features. The probability density of the eigenvalues should then follow
    /// a special densitiy function, described by the Marchenko-Pastur law.
    ///
    /// See also https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution
    #[test]
    fn test_marchenko_pastur() {
        // create random number generator
        let mut rng = Xoshiro256Plus::seed_from_u64(3);

        // generate normal distribution random data with N >> p
        let data = Array2::random_using((1000, 500), StandardNormal, &mut rng) / 1000f64.sqrt();

        let res = TruncatedSvd::new(data, Order::Largest)
            .decompose(500)
            .unwrap();

        let sv = res.values().mapv(|x: f64| x * x);

        // we have created a random spectrum and can apply the Marchenko-Pastur law
        // with variance 1 and p/n = 0.5
        let (a, b) = (
            1. * (1. - 0.5f64.sqrt()).powf(2.0),
            1. * (1. + 0.5f64.sqrt()).powf(2.0),
        );

        // check that the spectrum has correct boundaries
        assert_abs_diff_eq!(b, sv[0], epsilon = 0.1);
        assert_abs_diff_eq!(a, sv[sv.len() - 1], epsilon = 0.1);

        // estimate density empirical and compare with Marchenko-Pastur law
        let mut i = 0;
        'outer: for th in Array1::linspace(0.1, 2.8, 28).into_iter().rev() {
            let mut count = 0;
            while sv[i] >= *th {
                count += 1;
                i += 1;

                if i == sv.len() {
                    break 'outer;
                }
            }

            let x = th + 0.05;
            let mp_law = ((b - x) * (x - a)).sqrt() / std::f64::consts::PI / x;
            let empirical = count as f64 / 500. / ((2.8 - 0.1) / 28.);

            assert_abs_diff_eq!(mp_law, empirical, epsilon = 0.05);
        }
    }
}
