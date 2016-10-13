
extern crate ndarray;
extern crate ndarray_linalg as linalg;

use ndarray::prelude::*;
use linalg::Matrix;

fn main() {
    let a = arr2(&[[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]);
    println!("|a|_1 = {:?}", &a.norm_1());
    println!("|a|_1 = {:?}", &a.norm_i());
    println!("|a|_1 = {:?}", &a.norm_f());
}
