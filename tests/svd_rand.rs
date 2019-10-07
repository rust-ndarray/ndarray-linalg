use ndarray::*;
use ndarray_linalg::*;

#[cfg(feature = "sprs")]
use sprs::{CsMatBase, CsMatI};

fn test<A: SVDRand<U = Array2<f64>, Sigma = Array1<f64>, VT = Array2<f64>>>(a: &A, k: usize, answer: &Array2<f64>) {
    let s0 = answer.svd_dc(FlagSVD::Some).unwrap().1;
    let (u, s, vt) = a.svd_rand(k, None, None, None).unwrap();
    let u = u.unwrap();
    let vt = vt.unwrap();
    let mut sm = Array::zeros((k, k));
    for i in 0..k {
        sm[(i, i)] = s[i];
    }
    assert_close_l2!(&u.dot(&sm).dot(&vt), &answer, 0.2);
    assert_close_max!(&s, &s0.slice(s![..k]), 1e-7)
}

macro_rules! test_svd_rand_impl {
    ($m:expr, $n:expr, $k:expr) => {
        paste::item! {
            #[test]
            fn [<svd_rand_dense_ $m x $n _k $k>]() {
                // TODO(nlhepler): implement lower-rank matrices (w/ noise) for testing
                let a = random(($m, $n));
                test(&a, $k, &a);
            }

            #[cfg(feature = "sprs")]
            #[test]
            fn [<svd_rand_sparse_ $m x $n _k $k>]() {
                let a: Array2<f64> = random(($m, $n));
                let a: CsMatI<f64, usize> = CsMatBase::csr_from_dense(a.view(), 0.0);
                test(&a, $k, &a.to_dense());
            }
        }
    };
}

test_svd_rand_impl!(20, 20, 17);
test_svd_rand_impl!(25, 20, 17);
test_svd_rand_impl!(20, 25, 17);
