use ndarray::linalg::kron;
use ndarray::*;
use ndarray_linalg::expm::expm;
use ndarray_linalg::{Eig, OperationNorm};
use num_complex::{Complex64 as c64, ComplexFloat};
use rand::*;

/// Test matrix exponentiation expm by using an exact formula for exponentials of Pauli matrices.
/// These matrices are 1-sparse which allows for a different testing regime than the dense matrix
/// test.
#[test]
fn test_random_sparse_matrix() {
    let mut rng = rand::thread_rng();
    let num_qubits: u32 = 10;
    let dim = 2_usize.pow(num_qubits);
    let mut matrix = Array2::<c64>::eye(2);
    let _i = c64::new(0., 1.);
    let zero = c64::new(0., 0.);
    let pauli_x = array![[zero, c64::new(1., 0.)], [c64::new(1., 0.), zero]];
    let pauli_y = array![[zero, c64::new(0., -1.)], [c64::new(0., 1.), zero]];
    let pauli_z = array![[c64::new(1., 0.), zero], [zero, c64::new(-1., 0.)]];
    for n in 0..num_qubits {
        let pauli_matrix = match rng.gen_range::<i32, _>(0..=3) {
            0 => Array2::<c64>::eye(2),
            1 => pauli_x.clone(),
            2 => pauli_y.clone(),
            3 => pauli_z.clone(),
            _ => unreachable!(),
        };
        if n == 0 {
            matrix = matrix.dot(&pauli_matrix);
        } else {
            matrix = kron(&matrix, &pauli_matrix);
        }
    }
    // now check that this matrix squares to the identity as otherwise the exact value will be
    // incorrect.
    let matrix_squared = matrix.dot(&matrix);
    let diff = &matrix_squared - Array2::<c64>::eye(dim);
    assert!(diff.opnorm_one().unwrap() < 10. * (dim as f64) * f64::EPSILON);

    let theta = 1. * std::f64::consts::PI * rng.gen::<f64>();
    let scaled_matrix = matrix.clone() * c64::new(0., theta);
    let expm_computed = expm(&scaled_matrix).unwrap();
    let expm_expected = Array2::<c64>::eye(dim) * theta.cos() + c64::new(0., theta.sin()) * matrix;
    let comp_diff = &expm_expected - &expm_computed;

    let error = comp_diff.opnorm_one().unwrap() / f64::EPSILON;
    assert!(error <= 10.);
}

/// Test dense matrix exponentiation from random matrices. This works by constructing a random
/// unitary matrix as the eigenvectors and then random eigenvalues. We can then use the f64
/// exponentiation routine to compute the exponential of the diagonal eigenvalue matrix and then
/// multiply by the eigenvalues to compute the exponentiated matrix. This exact value is compared
/// to the expm calculation by looking at the average error per matrix entry. This test reveals an
/// error that scales with the dimension worse than competitors on the author's laptop (Mac M1 with
/// the Accelerate BLAS/LAPACK backend as of December 10th, 2022).
#[test]
fn test_low_dimension_random_dense_matrix() {
    let mut rng = rand::thread_rng();
    let dimensions = 100;
    let samps = 500;
    let scale = 1.;
    let mut results = Vec::new();
    let mut avg_entry_error = Vec::new();
    // Used to control what pade approximation is most likely to be used.
    // the smaller the norm the lower the degree used.
    for _ in 0..samps {
        // Sample a completely random matrix.
        let mut matrix: Array2<c64> = Array2::<c64>::ones((dimensions, dimensions).f());
        matrix.mapv_inplace(|_| c64::new(rng.gen::<f64>() * 1., rng.gen::<f64>() * 1.));

        // Make m positive semidefinite so it has orthonormal eigenvecs.
        matrix = matrix.dot(&matrix.t().map(|x| x.conj()));
        let (mut eigs, vecs) = matrix.eig().unwrap();
        let adjoint_vecs = vecs.t().clone().mapv(|x| x.conj());

        // Generate new random eigenvalues (complex, previously m had real eigenvals)
        // and a new matrix m
        eigs.mapv_inplace(|_| scale * c64::new(rng.gen::<f64>(), rng.gen::<f64>()));
        let new_matrix = vecs.dot(&Array2::from_diag(&eigs)).dot(&adjoint_vecs);

        // compute the exponentiated matrix by exponentiating the eigenvalues
        // and doing V e^Lambda V^\dagger
        eigs.mapv_inplace(|x| x.exp());
        let eigen_expm = vecs.dot(&Array2::from_diag(&eigs)).dot(&adjoint_vecs);

        // Compute the expm routine, compute error metrics for this sample
        let expm_comp = expm(&new_matrix).unwrap();
        let diff = &expm_comp - &eigen_expm;
        avg_entry_error.push({
            let tot = diff.map(|x| x.abs()).into_iter().sum::<f64>();
            tot / (dimensions * dimensions) as f64
        });
        results.push(diff.opnorm_one().unwrap());
    }

    // compute averages
    let avg: f64 = results.iter().sum::<f64>() / results.len() as f64;
    let avg_entry_diff = avg_entry_error.iter().sum::<f64>() / avg_entry_error.len() as f64;
    let _std: f64 = f64::powf(
        results.iter().map(|x| f64::powi(x - avg, 2)).sum::<f64>() / (results.len() - 1) as f64,
        0.5,
    );

    // This may fail at higher dimensions
    assert!(avg_entry_diff / f64::EPSILON <= 1000.)
}
