use ndarray::*;
use ndarray_linalg::*;

#[test]
fn generalize() {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let a: Array3<f64> = random_using((3, 2, 4).f(), &mut rng);
    let ans = a.clone();
    let a: Array3<f64> = convert::generalize(a);
    assert_eq!(a, ans);
}
