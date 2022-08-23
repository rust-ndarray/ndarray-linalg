mod tests {
    use crate::diagonal;

    /// Compares expm acting on a matrix with random eigenvalues (drawn from
    /// Gaussians) and with random eigenvectors (drawn from Haar distribution)
    /// to the exact answer. The exact answer is done by exponentiating each
    /// diagonal entry in the eigenvalue matrix before conjugating with the
    /// eigenvector matrices. In other words, let A = U D U^\dagger, then 
    /// because e^A = e^(U D U^\dagger) = U (e^D) U^dagger. We use expm
    /// to compute e^A and normal floating point exponentiation to compute e^D
    fn expm_test_gaussian_random_input(dim: usize) -> f32 {
        let D = diagonal::Diagonal::from([1,2,3,4,5]);
        let U = haar_random(dim);
        let F = floating_exp(D)
        let diff = expm(U D U.conj().T) - U F U.conj().T;
    }
}