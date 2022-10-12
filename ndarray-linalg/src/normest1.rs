use std::iter::{Zip, zip};

use ndarray::{concatenate, prelude::*};
use num_complex::ComplexFloat;
use rand::Rng;
// use rand::prelude::*;

const MAX_COLUMN_RESAMPLES: u32 = 10;

fn prepare_x_matrix(num_rows: usize, num_columns: usize) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    let mut output: Array2<f64> = Array::<f64, _>::zeros((num_rows, num_columns).f());
    output.mapv_inplace(|_| if rng.gen_bool(0.5) { 1.0 } else { -1.0 });
    // todo - check sizes?
    output.column_mut(0).fill(1.);
    ensure_no_parallel_columns(&mut output);
    output
}

fn is_column_parallel(index: usize, matrix: &Array2<f64>) -> bool {
    // TODO - no need to dot product entire matrix, just the preceeding columns
    let dot_prods = matrix.t().dot(&matrix.column(index));
    for ix in 0..index {
        if f64::abs(dot_prods[ix]) == matrix.nrows() as f64 {
            return true;
        }
    }
    false
}

fn ensure_column_not_parallel(matrix: &mut Array2<f64>, column_index: usize) {
    let mut rng = rand::thread_rng();
    for _ in 0..MAX_COLUMN_RESAMPLES {
        if is_column_parallel(column_index, matrix) {
            matrix.column_mut(column_index).mapv_inplace(|_| {
                if rng.gen_bool(0.5) {
                    1.0
                } else {
                    -1.0
                }
            });
        } else {
            break;
        }
    }
}

fn ensure_no_parallel_columns(mat: &mut Array2<f64>) {
    for col_ix in 0..mat.ncols() {
        ensure_column_not_parallel(mat, col_ix);
    }
}

/// Outputs true if every column of s_new is parallel to some column in s_old, false otherwise.
fn special_delivery(s_new: &Array2<f64>, s_old: &Array2<f64>) -> bool {
    let mut ret = true;
    let dots = s_old.t().dot(s_new);
    for col in dots.columns() {
        let mut is_col_good = false;
        for elem in col.iter() {
            if f64::abs(*elem) == col.len() as f64 {
                is_col_good = true;
            }
        }
        if is_col_good == false {
            ret = false;
            break;
        }
    }
    ret
}

fn ensure_new_s_matrix(s: &mut Array2<f64>, s_old: &Array2<f64>) {
    let mut big_matrix: Array2<f64> = concatenate(Axis(1), &[s_old.view(), s.view()]).unwrap();
    for col_ix in 0..s.ncols() {
        ensure_column_not_parallel(&mut big_matrix, s_old.ncols() + col_ix);
        for row_ix in 0..s.nrows() {
            s.column_mut(col_ix)[row_ix] = big_matrix.column(s_old.ncols() + col_ix)[row_ix];
        }
    }
}

fn check_index_history(indices: &Vec<usize>, index_history: &Vec<usize>, t: usize) -> bool {
    let mut ret = false;
    for chunk in index_history.chunks(t) {
        if *chunk == indices[0..t] {
            ret = true;
            break;
        }
    }
    ret
}

fn update_indices(indices: &mut Vec<usize>, index_history: &Vec<usize>, t: usize) {
    let mut unique_indices: Vec<usize> = Vec::with_capacity(t);
    for ix in indices.iter() {
        if index_history.contains(&ix) == false {
            unique_indices.push(*ix);
            if unique_indices.len() == t {
                break;
            }
        }
    }
    for ix in 0..unique_indices.len() {
        indices[ix] = unique_indices[ix];
    }
}

fn normest(A: &Array2<f64>, t: u32, itmax: u32) -> f64 {
    let mut est = 0.0;
    let mut best_index = 0;
    // Need to ensure that A is square
    let n = A.nrows();
    let mut x_matrix = prepare_x_matrix(n, t as usize);
    let mut index_history: Vec<usize> = Vec::new();
    let mut est_old = 0.0;
    let mut indices: Vec<usize> = (0..n).collect();
    let mut S_matrix: Array2<f64> = Array::<_, _>::zeros((n, t as usize).f());
    let mut S_old = S_matrix.clone();
    // Main loop of algorithm 2.4 in higham and tisseur
    for iteration in 0..itmax {
        let Y = A.dot(&x_matrix);
        let index_norm_pairs: Vec<(usize, f64)> = (0..Y.ncols())
            .into_iter()
            .map(|ix| (ix, Y.column(ix).map(|elem| f64::abs(*elem)).sum()))
            .collect();
        let mut ix_best = 0;
        let mut max_est = -1.0;
        for ix in 0..index_norm_pairs.len() {
            if index_norm_pairs[ix].1 > max_est {
                ix_best = ix;
                max_est = index_norm_pairs[ix].1;
            }
        }
        est = max_est;
        if est > est_old || iteration == 1 {
            best_index = ix_best;
        }
        if iteration > 1 && est < est_old {
            est = est_old;
            break;
        }
        S_old = S_matrix.clone();
        S_matrix = Y.mapv(|x| x.signum());
        if special_delivery(&S_matrix, &S_old) {
            break;
        }
        if t > 1 {
            ensure_new_s_matrix(&mut S_matrix, &S_old);
        }
        let z_matrix: Array2<f64> = A.t().dot(&S_matrix);
        let mut h: Array1<f64> = ndarray::Array::<f64, _>::zeros((z_matrix.nrows()).f());
        let mut max_h = f64::MIN;
        for row_ix in 0..z_matrix.nrows() {
            let mut max = f64::MIN;
            for col_ix in 0..z_matrix.ncols() {
                if z_matrix[[row_ix, col_ix]] > max {
                    max = z_matrix[[row_ix, col_ix]];
                }
            }
            h[[row_ix]] = max;
            if max > max_h {
                max_h = max;
            }
        }
        if iteration >= 2 && max_h == h[[best_index]] {
            break;
        }
        let mut zipped_pairs: Vec<(f64, usize)> = zip(h.to_vec(), indices.clone()).collect();
        zipped_pairs.sort_unstable_by(|a,b| b.0.partial_cmp(&a.0).unwrap());
        for ix in 0..zipped_pairs.len() {
            h[[ix]] = zipped_pairs[ix].0;
            indices[ix] = zipped_pairs[ix].1;
        }
        if t > 1 {
            if check_index_history(&indices, &index_history, t as usize) {
                break;
            }
            update_indices(&mut indices, &index_history, t as usize);
        }
        for ix in 0..(t as usize) {
            x_matrix.column_mut(ix).fill(0.0);
            x_matrix[[indices[ix], ix]] = 1.0;
            index_history.push(indices[ix]);
        }
    }
    est
}

mod tests {
    use crate::{
        ndarray::ShapeBuilder,
        normest1::{ensure_no_parallel_columns, is_column_parallel, prepare_x_matrix}, OperationNorm,
    };
    use ndarray::Array2;
    use rand::{thread_rng, Rng};

    use super::{special_delivery, normest};

    #[test]
    fn test_prep() {
        let n = 4;
        let t = 5;
        let mut x_mat: Array2<f64> = ndarray::Array::<f64, _>::zeros((n, t).f());
        println!("x_mat before");
        println!("{:?}", x_mat);
        // prepare_x_matrix(&mut x_mat);
        println!("x_mat after");
        println!("{:?}", x_mat);
        println!(
            "any parallel columns? {:}",
            is_column_parallel(x_mat.ncols() - 1, &x_mat)
        );
        println!("resampling columns");
        ensure_no_parallel_columns(&mut x_mat);
        println!("after resampling");
        println!("{:?}", x_mat);
    }

    #[test]
    fn test_one_norm() {
        let Y: Array2<f64> = array![[1., 2., 3., 4.], [1., 2., 3., 0.], [1., 2., 3., 0.]];
        let est = Y
            .columns()
            .into_iter()
            .map(|col| col.map(|x| f64::abs(*x)).sum())
            .reduce(f64::max)
            .unwrap();
        println!("est: {:}", est);
    }

    #[test]
    fn test_special_delivery() {
        let n = 100;
        let t = 4;
        let s_1: Array2<f64> = array![[-1., 1., 1.], [1., 1., 1.], [-1., -1., 1.]];
        let s_2: Array2<f64> = array![[-1., -1., 1.], [-1., -1., 1.], [-1., 1., -1.]];
        println!("special delivery: {:}", special_delivery(&s_2, &s_1));
    }

    #[test]
    fn test_normest() {
        let n = 100;
        let mut results = Vec::new();
        let t = 10;
        let itmax = 10;
        let mut mat: Array2<f64> = ndarray::Array::<f64, _>::zeros((n, n).f());
        let mut rng = rand::thread_rng();
        for _ in 0..1000 {
            mat.mapv_inplace(|_| rng.gen());
            results.push({
                normest(&mat, t as u32, itmax) / mat.opnorm_one().unwrap()
            });
        }
        println!("results average: {:}", results.iter().sum::<f64>() / results.len() as f64);
    }
}
