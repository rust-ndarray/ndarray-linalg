
extern crate rand;
extern crate ndarray;
extern crate ndarray_rand;
extern crate ndarray_linalg as linalg;

use linalg::*;

mod test_helper;

use test_helper::*;

#[test]
fn qr_square() {
    let qt = random_unitary(3);
    let rt = random_upper(3, 3);
    let a = qt.dot(&rt);
    let (q, r) = a.clone().qr().unwrap();
    println!("q = \n{:?}", q);
    println!("qt = \n{:?}", qt);
    println!("r = \n{:?}", r);
    println!("rt = \n{:?}", rt);
    all_close(r, rt);
    all_close(q, qt);
}

// #[test]
// fn qr_3x4() {
//     let r_dist = Range::new(0., 1.);
//     let a = Array::<f64, _>::random((3, 4), r_dist);
//     let (q, r) = a.clone().qr().unwrap();
//     println!("q = \n{:?}", q);
//     println!("r = \n{:?}", r);
//     all_close(a, q.dot(&r));
// }
//
// #[test]
// fn qr_4x3() {
//     let r_dist = Range::new(0., 1.);
//     let a = Array::<f64, _>::random((4, 3), r_dist);
//     let (q, r) = a.clone().qr().unwrap();
//     println!("q = \n{:?}", q);
//     println!("r = \n{:?}", r);
//     all_close(a, q.dot(&r));
// }
