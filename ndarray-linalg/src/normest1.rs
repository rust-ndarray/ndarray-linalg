use ndarray::prelude::*;
use num_complex::ComplexFloat;
use rand::Rng;
// use rand::prelude::*;

const MAX_COLUMN_RESAMPLES: u32 = 10;

fn prepare_x_matrix(num_rows: usize, num_columns: usize) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    let mut output: Array2<f64> = Array::<f64,_>::zeros((num_rows, num_columns).f());
    output.mapv_inplace(|_| if rng.gen_bool(0.5) {1.0} else {-1.0});
    // todo - check sizes?
    output.column_mut(0).fill(1.);
    resample_parallel_columns(&mut output);
    output
}

fn is_column_parallel(index: usize, matrix: &Array2<f64>) -> bool {
    let dot_prods = matrix.t().dot(&matrix.column(index));
    for ix in 0..index {
        if f64::abs(dot_prods[ix]) == (matrix.nrows() as f64) {
            return true;
        }
    }
    false
}

fn resample_parallel_columns(mat: &mut Array2<f64>) {
    let mut rng = rand::thread_rng();
    for col_ix in 0..mat.ncols() {
        // TODO- What if we hit the resample limit? should we warn or just ignore?
        for _ in 0..MAX_COLUMN_RESAMPLES {
            if is_column_parallel(col_ix, mat) {
                mat.column_mut(col_ix).mapv_inplace(|_| if rng.gen_bool(0.5) {1.0} else {-1.0});
            }
        }
    }
}

fn normest(A: Array2<f64>, t: u32, itmax: u32) -> f64 {
    let mut est = 0.;
    // Need to ensure that A is square
    let n = A.nrows();
    let mut x_matrix = prepare_x_matrix(n, t as usize);
    let mut index_history: Vec<usize> = Vec::new();
    let mut est_old = 0.;
    let mut ind: Array2<f64> = Array::<f64, _>::zeros((n, 1).f());
    let mut S_matrix: Array2<f64> = Array::<_,_>::zeros((n, t as usize).f());
    // Main loop of algorithm 2.4 in higham and tisseur
    for iteration in 0..=itmax {
        let Y = A.dot(&x_matrix);
        est = Y.columns().into_iter().map(|col| {
            col.iter().map(|elem| f64::abs(*elem)).sum()
        }).reduce(f64::max).unwrap()
    }
    est
}

mod tests {
    use ndarray::Array2;
    use crate::{ndarray::ShapeBuilder, normest1::{prepare_x_matrix, is_column_parallel, resample_parallel_columns}};

    #[test]
    fn test_prep() {
        let n = 4;
        let t = 5;
        let mut x_mat: Array2<f64> = ndarray::Array::<f64, _>::zeros((n,t).f());
        println!("x_mat before");
        println!("{:?}", x_mat);
        prepare_x_matrix(&mut x_mat);
        println!("x_mat after");
        println!("{:?}", x_mat);
        println!("any parallel columns? {:}", is_column_parallel(x_mat.ncols() - 1, &x_mat));
        println!("resampling columns");
        resample_parallel_columns(&mut x_mat);
        println!("after resampling");
        println!("{:?}", x_mat);
    }

    #[test]
    fn test_one_norm() {
        let Y: Array2<f64> = array![[1.,2.,3.,4.],[1.,2.,3.,4.],[1.,2.,3.,4.]];
        let est = Y.columns().into_iter().map(|col| {
            col.iter().map(|elem| f64::abs(*elem)).sum()
        }).reduce(f64::max).unwrap();
        println!("est: {:}", est);
    }
}