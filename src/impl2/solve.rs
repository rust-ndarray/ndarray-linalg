
use lapack::c;

use types::*;
use error::*;
use layout::Layout;

use super::into_result;

pub type Pivot = Vec<i32>;

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum Transpose {
    No = b'N',
    Transpose = b'T',
    Hermite = b'C',
}

pub trait Solve_: Sized {
    fn lu(Layout, a: &mut [Self]) -> Result<Pivot>;
    fn inv(Layout, a: &mut [Self], &Pivot) -> Result<()>;
    fn solve(Layout, Transpose, a: &[Self], &Pivot, b: &mut [Self]) -> Result<()>;
}

macro_rules! impl_solve {
    ($scalar:ty, $getrf:path, $getri:path, $getrs:path) => {

impl Solve_ for $scalar {
    fn lu(l: Layout, a: &mut [Self]) -> Result<Pivot> {
        let (row, col) = l.size();
        let k = ::std::cmp::min(row, col);
        let mut ipiv = vec![0; k as usize];
        let info = $getrf(l.lapacke_layout(), row, col, a, l.lda(), &mut ipiv);
        into_result(info, ipiv)
    }

    fn inv(l: Layout, a: &mut [Self], ipiv: &Pivot) -> Result<()> {
        let (n, _) = l.size();
        let info = $getri(l.lapacke_layout(), n, a, l.lda(), ipiv);
        into_result(info, ())
    }

    fn solve(l: Layout, t: Transpose, a: &[Self], ipiv: &Pivot, b: &mut [Self]) -> Result<()> {
        let (n, _) = l.size();
        let nrhs = 1;
        let ldb = 1;
        let info = $getrs(l.lapacke_layout(), t as u8, n, nrhs, a, l.lda(), ipiv, b, ldb);
        into_result(info, ())
    }
}

}} // impl_solve!

impl_solve!(f64, c::dgetrf, c::dgetri, c::dgetrs);
impl_solve!(f32, c::sgetrf, c::sgetri, c::sgetrs);
impl_solve!(c64, c::zgetrf, c::zgetri, c::zgetrs);
impl_solve!(c32, c::cgetrf, c::cgetri, c::cgetrs);
