//! Memory layout of matrices

pub type LDA = i32;
pub type LEN = i32;
pub type Col = i32;
pub type Row = i32;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MatrixLayout {
    C((Row, LDA)),
    F((Col, LDA)),
}

impl MatrixLayout {
    pub fn size(&self) -> (Row, Col) {
        match *self {
            MatrixLayout::C((row, lda)) => (row, lda),
            MatrixLayout::F((col, lda)) => (lda, col),
        }
    }

    pub fn resized(&self, row: Row, col: Col) -> MatrixLayout {
        match *self {
            MatrixLayout::C(_) => MatrixLayout::C((row, col)),
            MatrixLayout::F(_) => MatrixLayout::F((col, row)),
        }
    }

    pub fn lda(&self) -> LDA {
        std::cmp::max(
            1,
            match *self {
                MatrixLayout::C((_, lda)) | MatrixLayout::F((_, lda)) => lda,
            },
        )
    }

    pub fn len(&self) -> LEN {
        match *self {
            MatrixLayout::C((row, _)) => row,
            MatrixLayout::F((col, _)) => col,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn lapacke_layout(&self) -> lapacke::Layout {
        match *self {
            MatrixLayout::C(_) => lapacke::Layout::RowMajor,
            MatrixLayout::F(_) => lapacke::Layout::ColumnMajor,
        }
    }

    pub fn same_order(&self, other: &MatrixLayout) -> bool {
        self.lapacke_layout() == other.lapacke_layout()
    }

    pub fn toggle_order(&self) -> Self {
        match *self {
            MatrixLayout::C((row, col)) => MatrixLayout::F((col, row)),
            MatrixLayout::F((col, row)) => MatrixLayout::C((row, col)),
        }
    }
}
