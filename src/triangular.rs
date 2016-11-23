
use matrix::Matrix;
use square::SquareMatrix;

pub trait TriangularMatrix: Matrix + SquareMatrix {
    fn solve_upper(&self, Self::Vector) -> Self::Vector;
    fn solve_lower(&self, Self::Vector) -> Self::Vector;
}
