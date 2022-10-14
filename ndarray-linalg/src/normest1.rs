use std::iter::{zip, Zip};

use ndarray::{concatenate, prelude::*};
use num_complex::ComplexFloat;
use rand::Rng;

use crate::OperationNorm;
// use rand::prelude::*;

const MAX_COLUMN_RESAMPLES: u32 = 10;

fn prepare_x_matrix(num_rows: usize, num_columns: usize) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    let mut output: Array2<f64> = Array::<f64, _>::zeros((num_rows, num_columns).f());
    output.mapv_inplace(|_| if rng.gen_bool(0.5) { 1.0 } else { -1.0 });
    // todo - check sizes?
    output.column_mut(0).fill(1.);
    ensure_no_parallel_columns(&mut output);
    output / (num_rows as f64)
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
fn check_if_s_parallel_to_s_old(s_new: &Array2<f64>, s_old: &Array2<f64>) -> bool {
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

/// Resamples columns of s that are parallel to to any prior columns of s itself or to any column
/// of s_old.
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

/// Compute a lower bound of the 1-norm of a square matrix. The 1-norm is defined as
/// || A ||_1 = max_{1 <= j <= n} \sum_i |a_{i,j} |
/// In other words find the column with the largest sum over all rows (and take absolute value).
/// Note this is not equivalent to the induced 1-norm or the Schatten 1-norm.
/// We do not give the option of returning the v, w that maximize the resulting norm to keep the
/// function signature clean. This could be implemented in the future.
/// Panics if input matrix is non-square to avoid returning a Result<_,_>.
/// This is computed following Algorithm 2.4 of the paper "A Block Algorithm for Matrix 1-Norm
/// Estimation with an Application to 1-Norm Pseudospectra" by Nicholas J. Higham and
/// Francoise Tisseur (2000) SIAM J. Matrix Anal. Appl. Vol. 21, No. 4, pp. 1185-1201.
pub fn normest(input_matrix: &Array2<f64>, t: usize, itmax: u32) -> f64 {
    if input_matrix.is_square() == false {
        panic!("One Norm Estimation encountered non-square input matrix.");
    }
    if t < 1 {
        panic!("One Norm Estimation requires at least one column for estimation.");
    }
    if itmax < 2 {
        panic!("One Norm Estimation requires at least two iterations.");
    }
    let n = input_matrix.nrows();

    // This is worse than just computing the norm exactly, so just do that. Will panic
    // if opnorm_one() fails.
    if t > n {
        return input_matrix.opnorm_one().unwrap();
    }
    let mut est = 0.0;
    let mut best_index = 0;

    let mut x_matrix = prepare_x_matrix(n, t);
    let mut index_history: Vec<usize> = Vec::new();
    let mut est_old = 0.0;
    let mut indices: Vec<usize> = (0..n).collect();
    let mut s_matrix: Array2<f64> = Array::<_, _>::zeros((n, t).f());
    let mut s_old: Array2<f64> = Array::<_, _>::zeros((n, t).f());
    // Main loop of algorithm 2.4 in higham and tisseur
    for iteration in 0..itmax {
        // Y = AX
        let y = input_matrix.dot(&x_matrix);

        // est = max { ||Y(:, j)||_1 : j = 1:t}
        let index_norm_pairs: Vec<(usize, f64)> = (0..y.ncols())
            .into_iter()
            .map(|ix| (ix, y.column(ix).map(|elem| f64::abs(*elem)).sum()))
            .collect();
        let (maximizing_ix, max_est) = *index_norm_pairs
            .iter()
            .reduce(|x, y| if x.1 > y.1 { x } else { y })
            .unwrap();
        est = max_est;

        if est > est_old || iteration == 1 {
            best_index = indices[maximizing_ix];
        }

        // Section (1) of Alg. 2.4
        if iteration > 1 && est <= est_old {
            est = est_old;
            break;
        }

        est_old = est;
        s_old.assign(&s_matrix);
        // Is there a faster way to do this? I wanted to avoid creating a new matrix of signs
        // and then copying it into s_matrix, this was the first way I could think of doing so.
        for row_ix in 0..s_matrix.nrows() {
            for col_ix in 0..s_matrix.ncols() {
                s_matrix[[row_ix, col_ix]] = y[[row_ix, col_ix]].signum()
            }
        }

        // Section (2) of Alg. 2.4
        if check_if_s_parallel_to_s_old(&s_matrix, &s_old) {
            break;
        }
        if t > 1 {
            ensure_new_s_matrix(&mut s_matrix, &s_old);
        }

        // Section (3) of Alg. 2.4
        let z_matrix: Array2<f64> = input_matrix.t().dot(&s_matrix);
        let mut h: Vec<f64> = z_matrix
            .rows()
            .into_iter()
            .map(|row| *row.iter().reduce(|x, y| if x > y { x } else { y }).unwrap())
            .collect();
        let max_h = *h.iter().reduce(|x, y| if x > y { x } else { y }).unwrap();

        // Section (4) of Alg. 2.4
        if iteration >= 2 && max_h == h[best_index] {
            break;
        }
        let mut zipped_pairs: Vec<(f64, usize)> = zip(h.clone(), 0..n).collect();
        zipped_pairs.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        (h, indices) = zipped_pairs.into_iter().unzip();
        if t > 1 {
            // Section (5) of Alg. 2.4
            if check_index_history(&indices, &index_history, t) {
                break;
            }
            update_indices(&mut indices, &index_history, t);
        }
        for ix in 0..t {
            x_matrix.column_mut(ix).fill(0.0);
            x_matrix[[indices[ix], ix]] = 1.0;
            index_history.push(indices[ix]);
        }
    }
    // Would be Section (6) of Alg 2.4 if we returned the best index guess.
    est
}

mod tests {
    use crate::{
        ndarray::ShapeBuilder,
        normest1::{ensure_no_parallel_columns, is_column_parallel, prepare_x_matrix},
        OperationNorm, Inverse,
    };
    use ndarray::Array2;
    use rand::{thread_rng, Rng};

    use super::{check_if_s_parallel_to_s_old, normest};

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
    fn test_check_if_s_parallel_to_s_old() {
        let n = 100;
        let t = 4;
        let s_1: Array2<f64> = array![[-1., 1., 1.], [1., 1., 1.], [-1., -1., 1.]];
        let s_2: Array2<f64> = array![[-1., -1., 1.], [-1., -1., 1.], [-1., 1., -1.]];
        println!(
            "check_if_s_parallel_to_s_old: {:}",
            check_if_s_parallel_to_s_old(&s_2, &s_1)
        );
    }

    #[test]
    fn test_average_normest() {
        let n = 100;
        let mut differenial_error = Vec::new();
        let mut ratios = Vec::new();
        let t = 2;
        let itmax = 5;
        let mut mat: Array2<f64> = ndarray::Array::<f64, _>::zeros((n, n).f());
        let mut rng = rand::thread_rng();
        for _i in 0..5000 {
            mat.mapv_inplace(|_| rng.gen());
            mat.assign(&mat.inv().unwrap());
            let est = normest(&mat, t, itmax);
            let exp = mat.opnorm_one().unwrap();
            differenial_error.push((est - exp).abs() / exp);
            ratios.push(est / exp);
        }
        println!(
            "differential error average: {:}",
            differenial_error.iter().sum::<f64>() / differenial_error.len() as f64
        );
        println!(
            "ratio average: {:?}",
            ratios.iter().sum::<f64>() / ratios.len() as f64
        );
    }
}
