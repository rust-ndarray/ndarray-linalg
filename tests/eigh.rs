
extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::prelude::*;
use ndarray_linalg::prelude::*;

fn assert_almost_eq(a: f64, b: f64) {
    let rel_dev = (a - b).abs() / (a.abs() + b.abs());
    if rel_dev > 1.0e-7 {
        panic!("a={:?}, b={:?} are not almost equal", a, b);
    }
}


#[test]
fn eigen_vector_manual() {
    let a = arr2(&[[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]);
    let (e, vecs) = a.clone().eigh().unwrap();
    assert!(e.all_close(&arr1(&[2.0, 2.0, 5.0]), 1.0e-7));
    for (i, v) in vecs.axis_iter(Axis(1)).enumerate() {
        let av = a.dot(&v);
        let ev = v.mapv(|x| e[i] * x);
        assert!(av.all_close(&ev, 1.0e-7));
    }
}

#[test]
fn diagonalize() {
    let a = arr2(&[[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]);
    let (e, vecs) = a.clone().eigh().unwrap();
    let s = vecs.t().dot(&a).dot(&vecs);
    for i in 0..3 {
        assert_almost_eq(e[i], s[(i, i)]);
    }
}
