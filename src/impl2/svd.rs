//! Implement Operator norms for matrices

use lapack::c;

use types::*;
use error::*;
use layout::Layout;

#[repr(u8)]
pub enum FlagSVD {
    All = b'A',
    OverWrite = b'O',
    Separately = b'S',
    No = b'N',
}

pub trait SVD_: Sized {
    fn svd(Layout, u_flag: FlagSVD, v_flag: FlagSVD, a: &[Self]) -> Result<()>;
}
