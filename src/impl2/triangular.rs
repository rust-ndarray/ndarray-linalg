//! Implement linear solver and inverse matrix

use lapack::c;

use error::*;
use layout::Layout;
use super::{UPLO, Transpose, into_result};

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum Diag {
    Unit = b'U',
    NonUnit = b'N',
}

pub trait Triangular_: Sized {
    fn inv_triangular(l: Layout, UPLO, Diag, a: &mut [Self]) -> Result<()>;
    fn solve_triangular(al: Layout, bl: Layout, UPLO, Diag, a: &[Self], b: &mut [Self]) -> Result<()>;
}

impl Triangular_ for f64 {
    fn inv_triangular(l: Layout, uplo: UPLO, diag: Diag, a: &mut [Self]) -> Result<()> {
        let (n, _) = l.size();
        let lda = l.lda();
        let info = c::dtrtri(l.lapacke_layout(), uplo as u8, diag as u8, n, a, lda);
        into_result(info, ())
    }

    fn solve_triangular(al: Layout, bl: Layout, uplo: UPLO, diag: Diag, a: &[Self], mut b: &mut [Self]) -> Result<()> {
        let (n, _) = al.size();
        let lda = al.lda();
        let nrhs = bl.len();
        let ldb = bl.lda();
        let info = c::dtrtrs(al.lapacke_layout(),
                             uplo as u8,
                             Transpose::No as u8,
                             diag as u8,
                             n,
                             nrhs,
                             a,
                             lda,
                             &mut b,
                             ldb);
        into_result(info, ())
    }
}
