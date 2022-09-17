use ndarray::*;
use ndarray_linalg::*;

#[should_panic]
#[test]
fn solveh_shape_mismatch() {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let a: Array2<f64> = random_hpd_using(3, &mut rng);
    let b: Array1<f64> = random_using(2, &mut rng);
    let _ = a.solveh_into(b);
}

#[should_panic]
#[test]
fn factorizeh_solveh_shape_mismatch() {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let a: Array2<f64> = random_hpd_using(3, &mut rng);
    let b: Array1<f64> = random_using(2, &mut rng);
    let f = a.factorizeh_into().unwrap();
    let _ = f.solveh_into(b);
}

#[test]
fn solveh_random() {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let a: Array2<f64> = random_hpd_using(3, &mut rng);
    let x: Array1<f64> = random_using(3, &mut rng);
    let b = a.dot(&x);
    let y = a.solveh_into(b).unwrap();
    assert_close_l2!(&x, &y, 1e-7);

    let b = a.dot(&x);
    let f = a.factorizeh_into().unwrap();
    let y = f.solveh_into(b).unwrap();
    assert_close_l2!(&x, &y, 1e-7);
}

#[should_panic]
#[test]
fn solveh_t_shape_mismatch() {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let a: Array2<f64> = random_hpd_using(3, &mut rng).reversed_axes();
    let b: Array1<f64> = random_using(2, &mut rng);
    let _ = a.solveh_into(b);
}

#[should_panic]
#[test]
fn factorizeh_solveh_t_shape_mismatch() {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let a: Array2<f64> = random_hpd_using(3, &mut rng).reversed_axes();
    let b: Array1<f64> = random_using(2, &mut rng);
    let f = a.factorizeh_into().unwrap();
    let _ = f.solveh_into(b);
}

#[test]
fn solveh_random_t() {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let a: Array2<f64> = random_hpd_using(3, &mut rng).reversed_axes();
    let x: Array1<f64> = random_using(3, &mut rng);
    let b = a.dot(&x);
    let y = a.solveh_into(b).unwrap();
    assert_close_l2!(&x, &y, 1e-7);

    let b = a.dot(&x);
    let f = a.factorizeh_into().unwrap();
    let y = f.solveh_into(b).unwrap();
    assert_close_l2!(&x, &y, 1e-7);
}
