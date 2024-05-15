use std::num::NonZeroU8;

use anyhow::{anyhow, ensure, Context as _};

use crate::array::array_newtype;
use crate::bounded::impl_bounded_nonzero_uint;
use crate::hint::assert_unchecked;

const COL_NUM: u8 = 15;
const ROW_NUM: u8 = 12;

/// 盤面の列。左から右の順。
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Col(NonZeroU8);

impl_bounded_nonzero_uint!(Col, u8, 1, COL_NUM);

pub const COL_1: Col = unsafe { Col::from_inner_unchecked(1) };
pub const COL_2: Col = unsafe { Col::from_inner_unchecked(2) };
pub const COL_3: Col = unsafe { Col::from_inner_unchecked(3) };
pub const COL_4: Col = unsafe { Col::from_inner_unchecked(4) };
pub const COL_5: Col = unsafe { Col::from_inner_unchecked(5) };
pub const COL_6: Col = unsafe { Col::from_inner_unchecked(6) };
pub const COL_7: Col = unsafe { Col::from_inner_unchecked(7) };
pub const COL_8: Col = unsafe { Col::from_inner_unchecked(8) };
pub const COL_9: Col = unsafe { Col::from_inner_unchecked(9) };
pub const COL_10: Col = unsafe { Col::from_inner_unchecked(10) };
pub const COL_11: Col = unsafe { Col::from_inner_unchecked(11) };
pub const COL_12: Col = unsafe { Col::from_inner_unchecked(12) };
pub const COL_13: Col = unsafe { Col::from_inner_unchecked(13) };
pub const COL_14: Col = unsafe { Col::from_inner_unchecked(14) };
pub const COL_15: Col = unsafe { Col::from_inner_unchecked(15) };

impl Col {
    /// 左隣の列を返す。
    pub const fn prev(self) -> Option<Self> {
        if !matches!(self, Self::MIN) {
            Some(unsafe { self.prev_unchecked() })
        } else {
            None
        }
    }

    /// 左隣の列を返す。
    ///
    /// # Safety
    ///
    /// `self != Self::MIN` でなければならない。
    pub const unsafe fn prev_unchecked(self) -> Self {
        assert_unchecked!(!matches!(self, Self::MIN));

        Self::from_inner_unchecked(self.to_inner() - 1)
    }

    /// 右隣の列を返す。
    pub const fn next(self) -> Option<Self> {
        if !matches!(self, Self::MAX) {
            Some(unsafe { self.next_unchecked() })
        } else {
            None
        }
    }

    /// 右隣の列を返す。
    ///
    /// # Safety
    ///
    /// `self != Self::MAX` でなければならない。
    pub const unsafe fn next_unchecked(self) -> Self {
        assert_unchecked!(!matches!(self, Self::MAX));

        Self::from_inner_unchecked(self.to_inner() + 1)
    }
}

impl std::str::FromStr for Col {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let col: u8 = s
            .parse()
            .with_context(|| format!("Col のパースに失敗: '{s}'"))?;

        Col::from_inner(col).ok_or_else(|| anyhow!("Col の値が無効: {col}"))
    }
}

impl std::fmt::Display for Col {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// 盤面の行。下から上の順。
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Row(NonZeroU8);

impl_bounded_nonzero_uint!(Row, u8, 1, ROW_NUM);

pub const ROW_1: Row = unsafe { Row::from_inner_unchecked(1) };
pub const ROW_2: Row = unsafe { Row::from_inner_unchecked(2) };
pub const ROW_3: Row = unsafe { Row::from_inner_unchecked(3) };
pub const ROW_4: Row = unsafe { Row::from_inner_unchecked(4) };
pub const ROW_5: Row = unsafe { Row::from_inner_unchecked(5) };
pub const ROW_6: Row = unsafe { Row::from_inner_unchecked(6) };
pub const ROW_7: Row = unsafe { Row::from_inner_unchecked(7) };
pub const ROW_8: Row = unsafe { Row::from_inner_unchecked(8) };
pub const ROW_9: Row = unsafe { Row::from_inner_unchecked(9) };
pub const ROW_10: Row = unsafe { Row::from_inner_unchecked(10) };
pub const ROW_11: Row = unsafe { Row::from_inner_unchecked(11) };
pub const ROW_12: Row = unsafe { Row::from_inner_unchecked(12) };

impl std::str::FromStr for Row {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let col: u8 = s
            .parse()
            .with_context(|| format!("Row のパースに失敗: '{s}'"))?;

        Row::from_inner(col).ok_or_else(|| anyhow!("Row の値が無効: {col}"))
    }
}

impl std::fmt::Display for Row {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// 盤面のマス (column-major)。
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Square(NonZeroU8);

impl_bounded_nonzero_uint!(Square, u8, 1, COL_NUM * ROW_NUM);

impl Square {
    /// 列と行からマスを作る。
    pub const fn new(col: Col, row: Row) -> Self {
        let sq = ROW_NUM * (col.to_inner() - 1) + row.to_inner();

        unsafe { Self::from_inner_unchecked(sq) }
    }

    /// 属する列を返す。
    pub const fn col(self) -> Col {
        let col = 1 + (self.to_inner() - 1) / ROW_NUM;

        unsafe { Col::from_inner_unchecked(col) }
    }

    /// 属する行を返す。
    pub const fn row(self) -> Row {
        let row = 1 + (self.to_inner() - 1) % ROW_NUM;

        unsafe { Row::from_inner_unchecked(row) }
    }
}

impl std::str::FromStr for Square {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let fields: Vec<_> = s.split(',').collect();
        ensure!(fields.len() == 2, "Square のパースに失敗: '{s}'");

        let col: Col = fields[0]
            .parse()
            .with_context(|| format!("Square の列のパースに失敗: {}", fields[0]))?;
        let row: Row = fields[1]
            .parse()
            .with_context(|| format!("Square の行のパースに失敗: {}", fields[1]))?;

        Ok(Self::new(col, row))
    }
}

impl std::fmt::Display for Square {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{},{}", self.col(), self.row())
    }
}

array_newtype!(ColArray, Col);
array_newtype!(RowArray, Row);
array_newtype!(SquareArray, Square);

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_sq(s: impl AsRef<str>) -> Square {
        s.as_ref().parse().unwrap()
    }

    #[test]
    fn test_square_new() {
        for (col, row) in itertools::iproduct!(Col::all(), Row::all()) {
            let sq = Square::new(col, row);
            assert_eq!(sq.col(), col);
            assert_eq!(sq.row(), row);
        }
    }

    #[test]
    fn test_square_io() {
        for sq in Square::all() {
            let s = sq.to_string();
            assert_eq!(parse_sq(s), sq);
        }
    }
}
