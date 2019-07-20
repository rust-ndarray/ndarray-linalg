#![allow(non_snake_case)]
use ndarray::{stack, Array, Array1, ArrayBase, Axis, Data, Ix1, Ix2};
use ndarray_linalg::Solve;

/// The simple linear regression model is
///     y = bX + e  where e ~ N(0, sigma^2 * I)
/// In probabilistic terms this corresponds to
///     y - bX ~ N(0, sigma^2 * I)
///     y | X, b ~ N(bX, sigma^2 * I)
/// The loss for the model is simply the squared error between the model
/// predictions and the true values:
///     Loss = ||y - bX||^2
/// The maximum likelihood estimation for the model parameters `beta` can be computed
/// in closed form via the normal equation:
///     b = (X^T X)^{-1} X^T y
/// where (X^T X)^{-1} X^T is known as the pseudoinverse or Moore-Penrose inverse.
///
/// Adapted from: https://github.com/xinscrs/numpy-ml
pub struct LinearRegression {
    pub beta: Option<Array1<f64>>,
    fit_intercept: bool,
}

impl LinearRegression {
    pub fn new(fit_intercept: bool) -> LinearRegression {
        LinearRegression {
            beta: None,
            fit_intercept,
        }
    }

    pub fn fit<A, B>(&mut self, X: ArrayBase<A, Ix2>, y: ArrayBase<B, Ix1>)
    where
        A: Data<Elem = f64>,
        B: Data<Elem = f64>,
    {
        let (n_samples, _) = X.dim();

        // Check that our inputs have compatible shapes
        assert_eq!(y.dim(), n_samples);

        // If we are fitting the intercept, we need an additional column
        self.beta = if self.fit_intercept {
            let dummy_column: Array<f64, _> = Array::ones((n_samples, 1));
            let X = stack(Axis(1), &[dummy_column.view(), X.view()]).unwrap();
            Some(LinearRegression::solve_normal_equation(X, y))
        } else {
            Some(LinearRegression::solve_normal_equation(X, y))
        };
    }

    pub fn predict<A>(&self, X: &ArrayBase<A, Ix2>) -> Array1<f64>
        where
            A: Data<Elem = f64>,
    {
        let (n_samples, _) = X.dim();

        // If we are fitting the intercept, we need an additional column
        if self.fit_intercept {
            let dummy_column: Array<f64, _> = Array::ones((n_samples, 1));
            let X = stack(Axis(1), &[dummy_column.view(), X.view()]).unwrap();
            self._predict(&X)
        } else {
            self._predict(X)
        }
    }

    fn solve_normal_equation<A, B>(X: ArrayBase<A, Ix2>, y: ArrayBase<B, Ix1>) -> Array1<f64>
    where
        A: Data<Elem = f64>,
        B: Data<Elem = f64>,
    {
        let rhs = X.t().dot(&y);
        let linear_operator = X.t().dot(&X);
        linear_operator.solve_into(rhs).unwrap()
    }

    fn _predict<A>(&self, X: &ArrayBase<A, Ix2>) -> Array1<f64>
    where
        A: Data<Elem = f64>,
    {
        match &self.beta {
            None => panic!("The linear regression estimator has to be fitted first!"),
            Some(beta) => X.dot(beta),
        }
    }
}
