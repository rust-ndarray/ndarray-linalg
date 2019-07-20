#![allow(non_snake_case)]
use ndarray::{Array1, Array2, Array, Axis};
use ndarray_linalg::random;
use ndarray_stats::DeviationExt;
use ndarray_rand::RandomExt;
use rand::distributions::StandardNormal;

// Import LinearRegression from other file ("module") in this example
mod linear_regression;
use linear_regression::LinearRegression;

/// It returns a tuple: input data and the associated target variable.
///
/// The target variable is a linear function of the input, perturbed by gaussian noise.
fn get_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
    let shape = (n_samples, n_features);
    let noise: Array1<f64> = Array::random(n_samples, StandardNormal);

    let beta: Array1<f64> = random(n_features) * 10.;
    println!("Beta used to generate target variable: {:.3}", beta);

    let X: Array2<f64> = random(shape);
    let y: Array1<f64> = X.dot(&beta) + noise;
    (X, y)
}

pub fn main() {
    let n_train_samples = 5000;
    let n_test_samples = 1000;
    let n_features = 3;

    let (X, y) = get_data(n_train_samples + n_test_samples, n_features);
    let (X_train, X_test) = X.view().split_at(Axis(0), n_train_samples);
    let (y_train, y_test) = y.view().split_at(Axis(0), n_train_samples);

    let mut linear_regressor = LinearRegression::new(false);
    linear_regressor.fit(X_train, y_train);

    let test_predictions = linear_regressor.predict(&X_test);
    let mean_squared_error = test_predictions.mean_sq_err(&y_test.to_owned()).unwrap();
    println!(
        "Beta estimated from the training data: {:.3}",
        linear_regressor.beta.unwrap()
    );
    println!(
        "The fitted regressor has a mean squared error of {:.3}",
        mean_squared_error
    );
}
