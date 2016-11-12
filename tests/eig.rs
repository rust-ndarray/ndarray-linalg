
extern crate rand;
extern crate ndarray;
extern crate ndarray_rand;
extern crate ndarray_linalg;

use ndarray::prelude::*;
use ndarray_linalg::prelude::*;
use rand::distributions::*;
use ndarray_rand::RandomExt;

fn all_close(a: Array<f64, (Ix, Ix)>, b: Array<f64, (Ix, Ix)>) {
    if !a.all_close(&b, 1.0e-7) {
        panic!("\nTwo matrices are not equal:\na = \n{:?}\nb = \n{:?}\n",
               a,
               b);
    }
}

#[test]
fn eig_random() {
    let r_dist = Range::new(0., 1.);
    let a = Array::<f64, _>::random((3, 3), r_dist);
    let (w, vr) = a.clone().eig().unwrap();
    println!("w = \n{:?}", w);
    println!("vr = \n{:?}", vr);
    panic!("Manual Kill");
}
