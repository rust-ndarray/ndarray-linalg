use super::lobpcg::{lobpcg, LobpcgResult, Order};
use crate::{generate, Scalar};
use lax::Lapack;

///! Implements truncated eigenvalue decomposition
///
use ndarray::prelude::*;
use ndarray::stack;
use ndarray::ScalarOperand;
use num_traits::{Float, NumCast};

/// Truncated eigenproblem solver
///
/// This struct wraps the LOBPCG algorithm and provides convenient builder-pattern access to
/// parameter like maximal iteration, precision and constraint matrix. Furthermore it allows
/// conversion into a iterative solver where each iteration step yields a new eigenvalue/vector
/// pair.
pub struct TruncatedEig<A: Scalar> {
    order: Order,
    problem: Array2<A>,
    pub constraints: Option<Array2<A>>,
    preconditioner: Option<Array2<A>>,
    precision: f32,
    maxiter: usize,
}

impl<A: Float + Scalar + ScalarOperand + Lapack + PartialOrd + Default> TruncatedEig<A> {
    pub fn new(problem: Array2<A>, order: Order) -> TruncatedEig<A> {
        TruncatedEig {
            precision: 1e-5,
            maxiter: problem.len_of(Axis(0)) * 2,
            preconditioner: None,
            constraints: None,
            order,
            problem,
        }
    }

    pub fn precision(mut self, precision: f32) -> Self {
        self.precision = precision;

        self
    }

    pub fn maxiter(mut self, maxiter: usize) -> Self {
        self.maxiter = maxiter;

        self
    }

    pub fn orthogonal_to(mut self, constraints: Array2<A>) -> Self {
        self.constraints = Some(constraints);

        self
    }

    pub fn precondition_with(mut self, preconditioner: Array2<A>) -> Self {
        self.preconditioner = Some(preconditioner);

        self
    }

    // calculate the eigenvalues decompose
    pub fn decompose(&self, num: usize) -> LobpcgResult<A> {
        let x: Array2<f64> = generate::random((self.problem.len_of(Axis(0)), num));
        let x = x.mapv(|x| NumCast::from(x).unwrap());

        if let Some(ref preconditioner) = self.preconditioner {
            lobpcg(
                |y| self.problem.dot(&y),
                x,
                |mut y| y.assign(&preconditioner.dot(&y)),
                self.constraints.clone(),
                self.precision,
                self.maxiter,
                self.order.clone(),
            )
        } else {
            lobpcg(
                |y| self.problem.dot(&y),
                x,
                |_| {},
                self.constraints.clone(),
                self.precision,
                self.maxiter,
                self.order.clone(),
            )
        }
    }
}

impl<A: Float + Scalar + ScalarOperand + Lapack + PartialOrd + Default> IntoIterator
    for TruncatedEig<A>
{
    type Item = (Array1<A>, Array2<A>);
    type IntoIter = TruncatedEigIterator<A>;

    fn into_iter(self) -> TruncatedEigIterator<A> {
        TruncatedEigIterator {
            step_size: 1,
            remaining: self.problem.len_of(Axis(0)),
            eig: self,
        }
    }
}

/// Truncate eigenproblem iterator
///
/// This wraps a truncated eigenproblem and provides an iterator where each step yields a new
/// eigenvalue/vector pair. Useful for generating pairs until a certain condition is met.
pub struct TruncatedEigIterator<A: Scalar> {
    step_size: usize,
    remaining: usize,
    eig: TruncatedEig<A>,
}

impl<A: Float + Scalar + ScalarOperand + Lapack + PartialOrd + Default> Iterator
    for TruncatedEigIterator<A>
{
    type Item = (Array1<A>, Array2<A>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let step_size = usize::min(self.step_size, self.remaining);
        let res = self.eig.decompose(step_size);

        match res {
            LobpcgResult::Ok(vals, vecs, norms) | LobpcgResult::Err(vals, vecs, norms, _) => {
                // abort if any eigenproblem did not converge
                for r_norm in norms {
                    if r_norm > NumCast::from(0.1).unwrap() {
                        return None;
                    }
                }

                // add the new eigenvector to the internal constrain matrix
                let new_constraints = if let Some(ref constraints) = self.eig.constraints {
                    let eigvecs_arr: Vec<_> = constraints
                        .columns()
                        .into_iter()
                        .chain(vecs.columns().into_iter())
                        .collect();

                    stack(Axis(1), &eigvecs_arr).unwrap()
                } else {
                    vecs.clone()
                };

                self.eig.constraints = Some(new_constraints);
                self.remaining -= step_size;

                Some((vals, vecs))
            }
            LobpcgResult::NoResult(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Order;
    use super::TruncatedEig;
    use ndarray::{arr1, Array2};

    #[test]
    fn test_truncated_eig() {
        let diag = arr1(&[
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
            20.,
        ]);
        let a = Array2::from_diag(&diag);

        let teig = TruncatedEig::new(a, Order::Largest)
            .precision(1e-5)
            .maxiter(500);

        let res = teig
            .into_iter()
            .take(3)
            .flat_map(|x| x.0.to_vec())
            .collect::<Vec<_>>();
        let ground_truth = vec![20., 19., 18.];

        assert!(
            ground_truth
                .into_iter()
                .zip(res.into_iter())
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f64>()
                < 0.01
        );
    }
}
