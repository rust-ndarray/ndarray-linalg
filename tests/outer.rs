include!("header.rs");
use ndarray_linalg::vector::outer;

#[test]
fn outer_3x4() {
    let dist = RealNormal::<f64>::new(0.0, 1.0);
    let m = 3;
    let n = 4;
    let a = Array::random(m, dist);
    let b = Array::random(n, dist);
    println!("a = \n{:?}", &a);
    println!("b = \n{:?}", &b);
    let ab = outer(&a, &b);
    println!("ab = \n{:?}", &ab);
    for i in 0..m {
        for j in 0..n {
            ab[(i, j)].assert_close(a[i] * b[j], 1e-7);
        }
    }
}

#[test]
fn outer_4x3() {
    let dist = RealNormal::<f64>::new(0.0, 1.0);
    let m = 4;
    let n = 3;
    let a = Array::random(m, dist);
    let b = Array::random(n, dist);
    println!("a = \n{:?}", &a);
    println!("b = \n{:?}", &b);
    let ab = outer(&a, &b);
    println!("ab = \n{:?}", &ab);
    for i in 0..m {
        for j in 0..n {
            ab[(i, j)].assert_close(a[i] * b[j], 1e-7);
        }
    }
}
