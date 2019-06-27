use ndarray::*;
use ndarray_linalg::*;

fn test(a: &Array2<f64>, flag: FlagSVD) {
    let (m, n) = a.dim();
    let k = m.min(n);
    let answer = a.clone();
    println!("a = \n{:?}", a);
    let (u, s, vt): (_, Array1<_>, _) = a.svd_dc(flag).unwrap();
    let mut sm = match flag {
        FlagSVD::All => Array::zeros((m, n)),
        FlagSVD::Some => Array::zeros((k, k)),
        FlagSVD::None => {
            assert!(u.is_none());
            assert!(vt.is_none());
            return;
        },
    };
    let u: Array2<_> = u.unwrap();
    let vt: Array2<_> = vt.unwrap();
    println!("u = \n{:?}", &u);
    println!("s = \n{:?}", &s);
    println!("v = \n{:?}", &vt);
    for i in 0..k {
        sm[(i, i)] = s[i];
    }
    assert_close_l2!(&u.dot(&sm).dot(&vt), &answer, 1e-7);
}

macro_rules! test_svd_dc_impl {
    ($m:expr, $n:expr) => {
        paste::item! {
            #[test]
            fn [<svd_dc_full_ $m x $n>]() {
                let a = random(($m, $n));
                test(&a, FlagSVD::All);
            }

            #[test]
            fn [<svd_dc_some_ $m x $n>]() {
                let a = random(($m, $n));
                test(&a, FlagSVD::Some);
            }

            #[test]
            fn [<svd_dc_none_ $m x $n>]() {
                let a = random(($m, $n));
                test(&a, FlagSVD::None);
            }

            #[test]
            fn [<svd_dc_full_ $m x $n _t>]() {
                let a = random(($m, $n).f());
                test(&a, FlagSVD::All);
            }

            #[test]
            fn [<svd_dc_some_ $m x $n _t>]() {
                let a = random(($m, $n).f());
                test(&a, FlagSVD::Some);
            }

            #[test]
            fn [<svd_dc_none_ $m x $n _t>]() {
                let a = random(($m, $n).f());
                test(&a, FlagSVD::None);
            }
        }
    };
}

test_svd_dc_impl!(3, 3);
test_svd_dc_impl!(4, 3);
test_svd_dc_impl!(3, 4);
