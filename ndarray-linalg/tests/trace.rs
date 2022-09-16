use ndarray::*;
use ndarray_linalg::*;

#[test]
fn trace() {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let a: Array2<f64> = random_using((3, 3), &mut rng);
    assert_rclose!(a.trace().unwrap(), a[(0, 0)] + a[(1, 1)] + a[(2, 2)], 1e-7);
}
