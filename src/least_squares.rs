//! Least squares

// FIXME
// [ ] tests
// [ ] handle multiple RHS - this could be done via a trait type parameter
// [ ] use trait to call the right lapack function and get rid of macro

use ndarray::{Array, Array1, ArrayBase, Data, DataMut, Ix1, Ix2, s};

use crate::lapack::least_squares::*;
use crate::error::*;
use crate::layout::*;
use crate::types::*;

/// Result of LeastSquares
pub struct LeastSquaresResult<A: Scalar, I> {
    /// singular values
    pub singular_values: Array1<A::Real>,
    /// The solution matrix
    pub solution: Array<A, I>,
    /// The rank of the input matrix A
    pub rank: i32,
}

pub trait LeastSquaresSvd<S, A, I>
where
    S: Data<Elem = A>,
    A: Scalar + Lapack,
{
    fn least_squares(&self, rhs: &ArrayBase<S, I>) -> Result<LeastSquaresResult<A, I>>;
}

pub trait LeastSquaresSvdInto<S, A, I>
where
    S: Data<Elem = A>,
    A: Scalar + Lapack,
{
    fn least_squares_into(self, rhs: ArrayBase<S, I>) -> Result<LeastSquaresResult<A, I>>;
}

pub trait LeastSquaresSvdInPlace<S, A, I>
where
    S: Data<Elem = A>,
    A: Scalar + Lapack,
{
    fn least_squares_in_place(
        &mut self,
        rhs: &mut ArrayBase<S, I>,
    ) -> Result<LeastSquaresResult<A, I>>;
}

impl<A, S> LeastSquaresSvd<S, A, Ix1> for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack + LeastSquaresSvdDivideConquer_,
    S: Data<Elem = A>,
{
    fn least_squares(&self, rhs: &ArrayBase<S, Ix1>) -> Result<LeastSquaresResult<A, Ix1>> {
        let a = self.to_owned();
        let b = rhs.to_owned();
        a.least_squares_into(b)
    }
}

impl<A, S> LeastSquaresSvdInto<S, A, Ix1> for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack + LeastSquaresSvdDivideConquer_,
    S: DataMut<Elem = A>,
{
    fn least_squares_into(mut self, mut rhs: ArrayBase<S, Ix1>) -> Result<LeastSquaresResult<A, Ix1>> {
        self.least_squares_in_place(&mut rhs)
    }
}

impl<A, S> LeastSquaresSvdInPlace<S, A, Ix1> for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack + LeastSquaresSvdDivideConquer_,
    S: DataMut<Elem = A>,
{
    fn least_squares_in_place(
        &mut self,
        rhs: &mut ArrayBase<S, Ix1>,
    ) -> Result<LeastSquaresResult<A, Ix1>> {
        let a_layout = self.layout()?;
        let LeastSquaresOutput::<A> {
            singular_values,
            rank,
        } = unsafe {
            <A as LeastSquaresSvdDivideConquer_>::least_squares(
                a_layout,
                self.as_allocated_mut()?,
                rhs.as_slice_memory_order_mut()
                    .ok_or_else(|| LinalgError::MemoryNotCont)?,
            )?
        };

        let n = self.shape()[1];
        let solution = rhs.slice(s![0..n]).to_owned();
        Ok(LeastSquaresResult {
            solution,
            singular_values: Array::from_shape_vec((singular_values.len(),), singular_values)?,
            rank,
        })
    }
}

impl<A, S> LeastSquaresSvdInPlace<S, A, Ix2> for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack + LeastSquaresSvdDivideConquer_,
    S: DataMut<Elem = A>,
{
    fn least_squares_in_place(
        &mut self,
        rhs: &mut ArrayBase<S, Ix2>,
    ) -> Result<LeastSquaresResult<A, Ix2>> {
        let a_layout = self.layout()?;
        let rhs_layout = rhs.layout()?;
        let LeastSquaresOutput::<A> {
            singular_values,
            rank,
        } = unsafe {
            <A as LeastSquaresSvdDivideConquer_>::least_squares_nrhs(
                a_layout,
                self.as_allocated_mut()?,
                rhs_layout,
                rhs.as_allocated_mut()?
            )?
        };

        let solution = rhs.to_owned();
        Ok(LeastSquaresResult {
            solution,
            singular_values: Array::from_shape_vec((singular_values.len(),), singular_values)?,
            rank,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::AbsDiffEq;
    use ndarray::{Array1, Array2};

    #[test]
    fn netlib_lapack_example_for_dgels_1() {
        // https://www.netlib.org/lapack/lapacke.html#_calling_code_dgels_code
        let a: Array2<f64> = array![[1.,1.,1.],[2.,3.,4.], [3.,5.,2. ],[4.,2.,5.],[5.,4.,3.]];
        let b: Array1<f64> = array![-10.,12.,14.,16.,18.];
        let expected: Array1<f64> = array![2.,1.,1.];
        let result = a.least_squares(&b).unwrap();
        
        println!(" *****\nresult: {}\nexpected: {}\n *****", result.solution, expected);
        assert!(result.solution.abs_diff_eq(&expected, 1e-12));
    }

    #[test]
    fn netlib_lapack_example_for_dgels_2() {
        // https://www.netlib.org/lapack/lapacke.html#_calling_code_dgels_code
        let a: Array2<f64> = array![[1.,1.,1.],[2.,3.,4.], [3.,5.,2. ],[4.,2.,5.],[5.,4.,3.]];
        let b: Array1<f64> = array![-3., 14., 12., 16., 16.];
        let expected: Array1<f64> = array![1.,1.,2.];
        let result = a.least_squares(&b).unwrap();
        
        println!(" *****\nresult: {}\nexpected: {}\n *****", result.solution, expected);
        assert!(result.solution.abs_diff_eq(&expected, 1e-12));
    }

    #[test]
    fn netlib_lapack_example_for_dgels_nrhs() {
        // https://www.netlib.org/lapack/lapacke.html#_calling_code_dgels_code
        let mut a: Array2<f64> = array![[1.,1.,1.],[2.,3.,4.], [3.,5.,2. ],[4.,2.,5.],[5.,4.,3.]];
        let mut b: Array2<f64> = array![[-10.,12.,14.,16.,18.], [-3., 14., 12., 16., 16.]].t().to_owned();
        let expected: Array2<f64> = array![[2., 1.],[1., 1.], [1.,2.]];
        let result = a.least_squares_in_place(&mut b).unwrap();
        
        println!(" *****\nresult: {}\nexpected: {}\n *****", result.solution, expected);
        assert!(result.solution.abs_diff_eq(&expected, 1e-12));
    }

    // FIXME: test with multiple RHS
    // #[test]
    // fn netlib_lapack_example_for_dgels_2() {
    //     // https://www.netlib.org/lapack/lapacke.html#_calling_code_dgels_code
    //     let a: Array2<f64> = array![[1.,1.,1.],[2.,3.,4.], [3.,5.,2. ],[4.,2.,5.],[5.,4.,3.]];
    //     let b: Array1<f64> = array![-3., 14., 12., 16., 16.];
    //     let expected: Array1<f64> = array![1.,1.,2.];
    //     let result = a.least_squares(&b).unwrap();
        
    //     println!(" *****\nresult: {}\nexpected: {}\n *****", result.solution, expected);
    //     assert!(result.solution.abs_diff_eq(&expected, 1e-12));
//     ( -10 -3 )
//     (  12 14 )
// B = (  14 12 )
//     (  16 16 )
//     (  18 16 )
    // }
}
