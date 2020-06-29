//! Memory layout of matrices

pub type LDA = i32;
pub type LEN = i32;
pub type Col = i32;
pub type Row = i32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixLayout {
    C { row: i32, lda: i32 },
    F { col: i32, lda: i32 },
}

impl MatrixLayout {
    pub fn size(&self) -> (Row, Col) {
        match *self {
            MatrixLayout::C { row, lda } => (row, lda),
            MatrixLayout::F { col, lda } => (lda, col),
        }
    }

    pub fn resized(&self, row: Row, col: Col) -> MatrixLayout {
        match *self {
            MatrixLayout::C { .. } => MatrixLayout::C { row, lda: col },
            MatrixLayout::F { .. } => MatrixLayout::F { col, lda: row },
        }
    }

    pub fn lda(&self) -> LDA {
        std::cmp::max(
            1,
            match *self {
                MatrixLayout::C { lda, .. } | MatrixLayout::F { lda, .. } => lda,
            },
        )
    }

    pub fn len(&self) -> LEN {
        match *self {
            MatrixLayout::C { row, .. } => row,
            MatrixLayout::F { col, .. } => col,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn lapacke_layout(&self) -> lapacke::Layout {
        match *self {
            MatrixLayout::C { .. } => lapacke::Layout::RowMajor,
            MatrixLayout::F { .. } => lapacke::Layout::ColumnMajor,
        }
    }

    pub fn same_order(&self, other: &MatrixLayout) -> bool {
        self.lapacke_layout() == other.lapacke_layout()
    }

    pub fn toggle_order(&self) -> Self {
        match *self {
            MatrixLayout::C { row, lda } => MatrixLayout::F { lda: row, col: lda },
            MatrixLayout::F { col, lda } => MatrixLayout::C { row: lda, lda: col },
        }
    }
}
