use ndarray::*;
use ndarray_linalg::*;

macro_rules! cholesky {
    ($elem:ty, $rtol:expr) => {
        paste::item! {
            #[test]
            fn [<cholesky_ $elem>]() {
                let a_orig: Array2<$elem> = random_hpd(3);
                println!("a = \n{:?}", a_orig);

                let upper = a_orig.cholesky(UPLO::Upper).unwrap();
                assert_close_l2!(
                    &upper.t().mapv(|elem| elem.conj()).dot(&upper.view()),
                    &a_orig,
                    $rtol
                );

                let lower = a_orig.cholesky(UPLO::Lower).unwrap();
                assert_close_l2!(
                    &lower.dot(&lower.t().mapv(|elem| elem.conj())),
                    &a_orig,
                    $rtol
                );

                let a: Array2<$elem> = replicate(&a_orig);
                let upper = a.cholesky_into(UPLO::Upper).unwrap();
                assert_close_l2!(
                    &upper.t().mapv(|elem| elem.conj()).dot(&upper.view()),
                    &a_orig,
                    $rtol
                );

                let a: Array2<$elem> = replicate(&a_orig);
                let lower = a.cholesky_into(UPLO::Lower).unwrap();
                assert_close_l2!(
                    &lower.dot(&lower.t().mapv(|elem| elem.conj())),
                    &a_orig,
                    $rtol
                );

                let mut a: Array2<$elem> = replicate(&a_orig);
                {
                    let upper = a.cholesky_inplace(UPLO::Upper).unwrap();
                    assert_close_l2!(
                        &upper.t().mapv(|elem| elem.conj()).dot(&upper.view()),
                        &a_orig,
                        $rtol
                    );
                }
                assert_close_l2!(
                    &a.t().mapv(|elem| elem.conj()).dot(&upper.view()),
                    &a_orig,
                    $rtol
                );

                let mut a: Array2<$elem> = replicate(&a_orig);
                {
                    let lower = a.cholesky_inplace(UPLO::Lower).unwrap();
                    assert_close_l2!(
                        &lower.dot(&lower.t().mapv(|elem| elem.conj())),
                        &a_orig,
                        $rtol
                    );
                }
                assert_close_l2!(&a.dot(&lower.t().mapv(|elem| elem.conj())), &a_orig, $rtol);
            }
        } // paste::item
    };
}

cholesky!(f64, 1e-9);
cholesky!(f32, 1e-5);
cholesky!(c64, 1e-9);
cholesky!(c32, 1e-5);

macro_rules! cholesky_into_lower_upper {
    ($elem:ty, $rtol:expr) => {
        paste::item! {
            #[test]
            fn [<cholesky_into_lower_upper_ $elem>]() {
                let a: Array2<$elem> = random_hpd(3);
                println!("a = \n{:?}", a);
                let upper = a.cholesky(UPLO::Upper).unwrap();
                let fac_upper = a.factorizec(UPLO::Upper).unwrap();
                let fac_lower = a.factorizec(UPLO::Lower).unwrap();
                assert_close_l2!(&upper, &fac_lower.into_upper(), $rtol);
                assert_close_l2!(&upper, &fac_upper.into_upper(), $rtol);
                let lower = a.cholesky(UPLO::Lower).unwrap();
                let fac_upper = a.factorizec(UPLO::Upper).unwrap();
                let fac_lower = a.factorizec(UPLO::Lower).unwrap();
                assert_close_l2!(&lower, &fac_lower.into_lower(), $rtol);
                assert_close_l2!(&lower, &fac_upper.into_lower(), $rtol);
            }
        }
    };
}

cholesky_into_lower_upper!(f64, 1e-9);
cholesky_into_lower_upper!(f32, 1e-5);
cholesky_into_lower_upper!(c64, 1e-9);
cholesky_into_lower_upper!(c32, 1e-5);

macro_rules! cholesky_into_inverse {
    ($elem:ty, $rtol:expr) => {
        paste::item! {
            #[test]
            fn [<cholesky_inverse_ $elem>]() {
                let a: Array2<$elem> = random_hpd(3);
                println!("a = \n{:?}", a);
                let inv = a.invc().unwrap();
                assert_close_l2!(&a.dot(&inv), &Array2::eye(3), $rtol);
                let inv_into: Array2<$elem> = replicate(&a).invc_into().unwrap();
                assert_close_l2!(&a.dot(&inv_into), &Array2::eye(3), $rtol);
                let inv_upper = a.factorizec(UPLO::Upper).unwrap().invc().unwrap();
                assert_close_l2!(&a.dot(&inv_upper), &Array2::eye(3), $rtol);
                let inv_upper_into = a.factorizec(UPLO::Upper).unwrap().invc_into().unwrap();
                assert_close_l2!(&a.dot(&inv_upper_into), &Array2::eye(3), $rtol);
                let inv_lower = a.factorizec(UPLO::Lower).unwrap().invc().unwrap();
                assert_close_l2!(&a.dot(&inv_lower), &Array2::eye(3), $rtol);
                let inv_lower_into = a.factorizec(UPLO::Lower).unwrap().invc_into().unwrap();
                assert_close_l2!(&a.dot(&inv_lower_into), &Array2::eye(3), $rtol);
            }
        }
    };
}
cholesky_into_inverse!(f64, 1e-9);
cholesky_into_inverse!(f32, 1e-3);
cholesky_into_inverse!(c64, 1e-9);
cholesky_into_inverse!(c32, 1e-3);

macro_rules! cholesky_det {
    ($elem:ty, $atol:expr) => {
        paste::item! {
            #[test]
            fn [<cholesky_det_ $elem>]() {
                let a: Array2<$elem> = random_hpd(3);
                println!("a = \n{:?}", a);
                let ln_det = a
                    .eigvalsh(UPLO::Upper)
                    .unwrap()
                    .mapv(|elem| elem.ln())
                    .sum();
                let det = ln_det.exp();
                assert_aclose!(a.factorizec(UPLO::Upper).unwrap().detc(), det, $atol);
                assert_aclose!(a.factorizec(UPLO::Upper).unwrap().ln_detc(), ln_det, $atol);
                assert_aclose!(a.factorizec(UPLO::Lower).unwrap().detc_into(), det, $atol);
                assert_aclose!(
                    a.factorizec(UPLO::Lower).unwrap().ln_detc_into(),
                    ln_det,
                    $atol
                );
                assert_aclose!(a.detc().unwrap(), det, $atol);
                assert_aclose!(a.ln_detc().unwrap(), ln_det, $atol);
                assert_aclose!(a.clone().detc_into().unwrap(), det, $atol);
                assert_aclose!(a.ln_detc_into().unwrap(), ln_det, $atol);
            }
        }
    };
}
cholesky_det!(f64, 1e-9);
cholesky_det!(f32, 1e-3);
cholesky_det!(c64, 1e-9);
cholesky_det!(c32, 1e-3);

macro_rules! cholesky_solve {
    ($elem:ty, $rtol:expr) => {
        paste::item! {
            #[test]
            fn [<cholesky_solve_ $elem>]() {
                let a: Array2<$elem> = random_hpd(3);
                let x: Array1<$elem> = random(3);
                let b = a.dot(&x);
                println!("a = \n{:?}", a);
                println!("x = \n{:?}", x);
                assert_close_l2!(&a.solvec(&b).unwrap(), &x, $rtol);
                assert_close_l2!(&a.solvec_into(b.clone()).unwrap(), &x, $rtol);
                assert_close_l2!(&a.solvec_inplace(&mut b.clone()).unwrap(), &x, $rtol);
                assert_close_l2!(
                    &a.factorizec(UPLO::Upper).unwrap().solvec(&b).unwrap(),
                    &x,
                    $rtol
                );
                assert_close_l2!(
                    &a.factorizec(UPLO::Lower).unwrap().solvec(&b).unwrap(),
                    &x,
                    $rtol
                );
                assert_close_l2!(
                    &a.factorizec(UPLO::Upper)
                        .unwrap()
                        .solvec_into(b.clone())
                        .unwrap(),
                    &x,
                    $rtol
                );
                assert_close_l2!(
                    &a.factorizec(UPLO::Lower)
                        .unwrap()
                        .solvec_into(b.clone())
                        .unwrap(),
                    &x,
                    $rtol
                );
                assert_close_l2!(
                    &a.factorizec(UPLO::Upper)
                        .unwrap()
                        .solvec_inplace(&mut b.clone())
                        .unwrap(),
                    &x,
                    $rtol
                );
                assert_close_l2!(
                    &a.factorizec(UPLO::Lower)
                        .unwrap()
                        .solvec_inplace(&mut b.clone())
                        .unwrap(),
                    &x,
                    $rtol
                );
            }
        }
    };
}
cholesky_solve!(f64, 1e-9);
cholesky_solve!(f32, 1e-3);
cholesky_solve!(c64, 1e-9);
cholesky_solve!(c32, 1e-3);
