
pub use vector::Norm;
pub use matrix::Matrix;
pub use square::SquareMatrix;
pub use hermite::HermiteMatrix;
pub use triangular::{SolveTriangular, drop_lower, drop_upper};
pub use util::{all_close_l1, all_close_l2, all_close_max};
