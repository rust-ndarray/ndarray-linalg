use ndarray::*;
use ndarray_linalg::*;

fn test1d<A, Sa, Sb>(uplo: UPLO, a: &ArrayBase<Sa, Ix2>, b: &ArrayBase<Sb, Ix1>, tol: A::Real)
where
    A: Scalar + Lapack,
    Sa: Data<Elem = A>,
    Sb: DataMut<Elem = A> + DataOwned,
{
    println!("a = {:?}", a);
    println!("b = {:?}", b);
    let x = a.solve_triangular(uplo, Diag::NonUnit, b).unwrap();
    println!("x = {:?}", &x);
    let b_ = a.dot(&x);
    println!("Ax = {:?}", &b_);
    assert_close_l2!(&b_, b, tol);
}

fn test2d<A, Sa, Sb>(uplo: UPLO, a: &ArrayBase<Sa, Ix2>, b: &ArrayBase<Sb, Ix2>, tol: A::Real)
where
    A: Scalar + Lapack,
    Sa: Data<Elem = A>,
    Sb: DataMut<Elem = A> + DataOwned + Data + RawDataClone,
{
    println!("a = {:?}", a);
    println!("b = {:?}", b);
    let x = a.solve_triangular(uplo, Diag::NonUnit, b).unwrap();
    println!("x = {:?}", &x);
    let b_ = a.dot(&x);
    println!("Ax = {:?}", &b_);
    assert_close_l2!(&b_, b, tol);
}

#[test]
fn triangular_1d_upper() {
    let n = 3;
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let b: Array1<f64> = random_using(n, &mut rng);
    let a: Array2<f64> = random_using((n, n), &mut rng).into_triangular(UPLO::Upper);
    test1d(UPLO::Upper, &a, &b, 1e-7);
}

#[test]
fn triangular_1d_lower() {
    let n = 3;
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let b: Array1<f64> = random_using(n, &mut rng);
    let a: Array2<f64> = random_using((n, n), &mut rng).into_triangular(UPLO::Lower);
    test1d(UPLO::Lower, &a, &b, 1e-7);
}

#[test]
fn triangular_1d_upper_t() {
    let n = 3;
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let b: Array1<f64> = random_using(n, &mut rng);
    let a: Array2<f64> = random_using((n, n).f(), &mut rng).into_triangular(UPLO::Upper);
    test1d(UPLO::Upper, &a, &b, 1e-7);
}

#[test]
fn triangular_1d_lower_t() {
    let n = 3;
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let b: Array1<f64> = random_using(n, &mut rng);
    let a: Array2<f64> = random_using((n, n).f(), &mut rng).into_triangular(UPLO::Lower);
    test1d(UPLO::Lower, &a, &b, 1e-7);
}

#[test]
fn triangular_2d_upper() {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let b: Array2<f64> = random_using((3, 4), &mut rng);
    let a: Array2<f64> = random_using((3, 3), &mut rng).into_triangular(UPLO::Upper);
    test2d(UPLO::Upper, &a, &b, 1e-7);
}

#[test]
fn triangular_2d_lower() {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let b: Array2<f64> = random_using((3, 4), &mut rng);
    let a: Array2<f64> = random_using((3, 3), &mut rng).into_triangular(UPLO::Lower);
    test2d(UPLO::Lower, &a, &b, 1e-7);
}

#[test]
fn triangular_2d_lower_t() {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let b: Array2<f64> = random_using((3, 4), &mut rng);
    let a: Array2<f64> = random_using((3, 3).f(), &mut rng).into_triangular(UPLO::Lower);
    test2d(UPLO::Lower, &a, &b, 1e-7);
}

#[test]
fn triangular_2d_upper_t() {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let b: Array2<f64> = random_using((3, 4), &mut rng);
    let a: Array2<f64> = random_using((3, 3).f(), &mut rng).into_triangular(UPLO::Upper);
    test2d(UPLO::Upper, &a, &b, 1e-7);
}

#[test]
fn triangular_2d_upper_bt() {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let b: Array2<f64> = random_using((3, 4).f(), &mut rng);
    let a: Array2<f64> = random_using((3, 3), &mut rng).into_triangular(UPLO::Upper);
    test2d(UPLO::Upper, &a, &b, 1e-7);
}

#[test]
fn triangular_2d_lower_bt() {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let b: Array2<f64> = random_using((3, 4).f(), &mut rng);
    let a: Array2<f64> = random_using((3, 3), &mut rng).into_triangular(UPLO::Lower);
    test2d(UPLO::Lower, &a, &b, 1e-7);
}

#[test]
fn triangular_2d_lower_t_bt() {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let b: Array2<f64> = random_using((3, 4).f(), &mut rng);
    let a: Array2<f64> = random_using((3, 3).f(), &mut rng).into_triangular(UPLO::Lower);
    test2d(UPLO::Lower, &a, &b, 1e-7);
}

#[test]
fn triangular_2d_upper_t_bt() {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let b: Array2<f64> = random_using((3, 4).f(), &mut rng);
    let a: Array2<f64> = random_using((3, 3).f(), &mut rng).into_triangular(UPLO::Upper);
    test2d(UPLO::Upper, &a, &b, 1e-7);
}
