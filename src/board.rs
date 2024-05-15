//! 盤面関連。

use std::fmt::Write as _;

use anyhow::{bail, ensure};

use crate::bitop;
use crate::hint::assert_unchecked;
use crate::piece::Piece;
use crate::square::{Col, ColArray, Row, RowArray, Square};

type BitColT = u64;

/// bitboard の列。
///
/// 1 マス 3bit。
/// メソッドに渡すマスの値は `0b111` 以下でなければならない。
#[repr(transparent)]
#[derive(Clone, Copy, Eq, PartialEq)]
struct BitCol(BitColT);

impl Default for BitCol {
    fn default() -> Self {
        Self::zero()
    }
}

impl BitCol {
    const fn zero() -> Self {
        Self(0)
    }

    const fn new(inner: BitColT) -> Self {
        Self(inner)
    }

    /// 全マスの値が `value` であるような `BitCol` を返す。
    const fn broadcast(value: u8) -> Self {
        const UNIT: BitColT = 0b001_001_001_001_001_001_001_001_001_001_001_001;

        unsafe { assert_unchecked!(Self::value_is_ok(value)) }

        Self::new(UNIT * value as BitColT)
    }

    fn from_pieces(pieces: &RowArray<Piece>) -> Self {
        let mut bc = Self::zero();

        for (row, piece) in pieces.enumerate() {
            bc.set(row, piece.to_inner());
        }

        bc
    }

    const fn inner(self) -> BitColT {
        self.0
    }

    const fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// 指定した行のマスの値を返す。
    const fn get(self, row: Row) -> u8 {
        ((self.0 >> (3 * row.to_index())) & 0b111) as u8
    }

    /// 指定した行のマスの値を設定する。
    fn set(&mut self, row: Row, value: u8) {
        unsafe { assert_unchecked!(Self::value_is_ok(value)) }

        self.0 &= !(0b111 << (3 * row.to_index()));
        self.0 |= BitColT::from(value) << (3 * row.to_index());
    }

    fn iter(
        self,
    ) -> impl DoubleEndedIterator<Item = u8> + ExactSizeIterator + std::iter::FusedIterator + Clone
    {
        self.enumerate().map(|(_, value)| value)
    }

    fn enumerate(
        self,
    ) -> impl DoubleEndedIterator<Item = (Row, u8)> + ExactSizeIterator + std::iter::FusedIterator + Clone
    {
        Row::all().map(move |row| (row, self.get(row)))
    }

    const fn value_is_ok(value: u8) -> bool {
        value <= 0b111
    }
}

impl std::ops::BitAnd for BitCol {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl std::ops::BitOr for BitCol {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitXor for BitCol {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

impl std::ops::BitAndAssign for BitCol {
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl std::ops::BitOrAssign for BitCol {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl std::ops::BitXorAssign for BitCol {
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

impl std::fmt::Debug for BitCol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("0b")?;

        for row in Row::all().rev() {
            let value = self.get(row);
            write!(f, "_{value:03b}")?;
        }

        Ok(())
    }
}

/// 盤面。
///
/// `BitCol` を `Col::NUM` 個持っており、常に左詰めされている。
///
/// `BitCol` のマスの値は 0 が空白、`1..=5` が各駒種を表す。
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Board {
    bcs: ColArray<BitCol>,
    width_remain: u32,
}

impl Board {
    const CHAR_BLANK: char = '.';

    /// `Board` を生成する。デバッグモードでは不変条件のチェックも行う。
    fn new(bcs: ColArray<BitCol>, width_remain: u32) -> Self {
        unsafe { assert_unchecked!(width_remain as usize <= Col::NUM) }

        debug_assert!(
            bcs.as_array()
                .iter()
                .copied()
                .all(|bc| bc.iter().all(|value| value <= Piece::MAX_VALUE)),
            "Board のマスの値が正しくない"
        );

        debug_assert!(
            bcs.as_array()[..width_remain as usize]
                .iter()
                .copied()
                .all(|bc| !bc.is_zero()),
            "Board の左端 width_remain 列に空の列がある"
        );

        debug_assert!(
            bcs.as_array()[width_remain as usize..]
                .iter()
                .copied()
                .all(BitCol::is_zero),
            "Board の左端 width_remain 列以外に空でない列がある"
        );

        Self { bcs, width_remain }
    }

    /// 空の盤面を返す。
    pub fn empty() -> Self {
        Self::new(ColArray::default(), 0)
    }

    /// Piece の 2 次元配列 (column-major) から盤面を生成する。
    /// ゲーム内の再生成判定は考慮しない。
    pub fn from_piece_arrays(arrays: &ColArray<RowArray<Piece>>) -> Self {
        let bcs = ColArray::from_fn(|col| BitCol::from_pieces(&arrays[col]));

        Self::new(bcs, Col::NUM as u32)
    }

    /// 指定したマスの駒を返す。
    pub fn get(&self, sq: Square) -> Option<Piece> {
        let value = self.bcs[sq.col()].get(sq.row());
        unsafe { assert_unchecked!(value <= Piece::MAX_VALUE) }

        Piece::from_inner(value)
    }

    /// 盤面が空かどうかを返す。
    pub fn is_empty(&self) -> bool {
        self.width_remain == 0
    }

    /// 空でない列数を返す。
    pub fn width_remain(&self) -> u32 {
        self.width_remain
    }

    /// 空でない列を昇順で列挙する。
    pub fn nonempty_cols(
        &self,
    ) -> impl ExactSizeIterator<Item = Col> + std::iter::FusedIterator + Clone {
        (0..self.width_remain as u8).map(|i| unsafe { Col::from_inner_unchecked(1 + i) })
    }

    /// 指定した駒種の数を返す。
    pub fn piece_count(&self, piece: Piece) -> u32 {
        self.piece_mask(piece).square_count()
    }

    /// 駒の総数を返す。
    pub fn piece_count_total(&self) -> u32 {
        self.nonempty_cols()
            .map(|col| {
                let bc = self.bcs[col].0;
                let bc = (bc | (bc >> 1) | (bc >> 2)) & BitCol::broadcast(0b001).0;
                bc.count_ones()
            })
            .sum()
    }

    /// 指定した駒のみからなる盤面マスクを返す。
    pub fn piece_mask(&self, piece: Piece) -> MaskBoard {
        // まず全体を piece の内部値で埋めた盤面との XOR をとる。
        // すると、指定した駒のあるマスのみが 0b000 である盤面が得られる。
        //
        // 他のマスは 0b000 でない値になっているので、適当にシフトと AND を用いて値を 0b001 に統一する。
        // そして、全マスに対して 0b001 を XOR すれば求めるマスクが得られる。
        //
        // 実際には各列について上記を個別に行う。

        let filled = BitCol::broadcast(piece.to_inner());

        let mut bcs = ColArray::<BitCol>::default();
        let mut col_mask = 0;

        for col in self.nonempty_cols() {
            let bc = (self.bcs[col] ^ filled).0;
            let bc = (bc | (bc >> 1) | (bc >> 2)) & BitCol::broadcast(0b001).0;
            let bc = BitCol::new(bc) ^ BitCol::broadcast(0b001);
            bcs[col] = bc;
            if !bc.is_zero() {
                col_mask |= 1 << col.to_index();
            }
        }

        MaskBoard::new(bcs, col_mask)
    }

    /// 各駒種について連結成分を列挙する。孤立駒も含むことに注意。
    pub fn piece_components(
        &self,
    ) -> impl std::iter::FusedIterator<Item = (Piece, MaskBoard)> + Clone + '_ {
        Piece::all().flat_map(|piece| {
            self.piece_mask(piece)
                .components()
                .map(move |comp| (piece, comp))
        })
    }

    /// 合法手があるかどうかを返す。
    pub fn has_action(&self) -> bool {
        // 盤面が空なら明らかに合法手はない。
        // そうでない場合、各駒種についてマスクを求め、
        // それを上下方向/左右方向にずらしたとき重なる部分があるかどうかを見ればよい。

        if self.is_empty() {
            return false;
        }

        Piece::all().any(|piece| {
            let mb = self.piece_mask(piece);

            mb.nonempty_cols().any(|col| {
                let bc = mb.bcs[col].0;
                if (bc & (bc >> 3)) != 0 {
                    return true;
                }
                if let Some(col_prev) = col.prev() {
                    let bc_prev = mb.bcs[col_prev].0;
                    if (bc & bc_prev) != 0 {
                        return true;
                    }
                }
                false
            })
        })
    }

    /// 与えられた盤面マスク内の全ての駒を消し、その結果を返す。
    pub fn erase(&self, mb: &MaskBoard) -> Self {
        // mb の各マスの値は 0b000, 0b001 の 2 値だが、0b111 を掛けることで 0b000, 0b111 の 2 値に変換できる。
        // これの NOT をマスクとして PEXT を行えばよい。
        //
        // 列の詰め直しは愚直に行う。この操作の頻度は低いのでさほど問題にはならないだろう。

        let mut bcs = self.bcs.clone();
        let mut erased_col_mask = 0;
        for col in mb.nonempty_cols() {
            let mask = !(mb.bcs[col].0 * 0b111);
            bcs[col] = BitCol::new(bitop::u64_pext(bcs[col].0, mask));
            if bcs[col].is_zero() {
                erased_col_mask |= 1 << col.to_index();
            }
        }

        let (bcs, width_remain) = if erased_col_mask == 0 {
            (bcs, self.width_remain)
        } else {
            let mut res = ColArray::<BitCol>::default();
            let mut width_remain = 0;
            for col in self.nonempty_cols() {
                if (erased_col_mask & (1 << col.to_index())) != 0 {
                    continue;
                }
                let col_out = unsafe { Col::from_inner_unchecked(1 + width_remain) };
                res[col_out] = bcs[col];
                width_remain += 1;
            }
            (res, u32::from(width_remain))
        };

        Self::new(bcs, width_remain)
    }

    /// `self` と `other` で値が異なるマスの集合を表す盤面マスクを返す。
    pub fn xor_mask(&self, other: &Self) -> MaskBoard {
        let mut bcs = ColArray::<BitCol>::default();
        let mut col_mask = 0;

        let cols = if self.width_remain >= other.width_remain {
            self.nonempty_cols()
        } else {
            other.nonempty_cols()
        };
        for col in cols {
            let bc = (self.bcs[col] ^ other.bcs[col]).0;
            let bc = (bc | (bc >> 1) | (bc >> 2)) & BitCol::broadcast(0b001).0;
            bcs[col] = BitCol::new(bc);
            if !bcs[col].is_zero() {
                col_mask |= 1 << col.to_index();
            }
        }

        MaskBoard::new(bcs, col_mask)
    }
}

impl std::str::FromStr for Board {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lines: Vec<_> = s.lines().collect();
        ensure!(
            lines.len() == Row::NUM,
            "盤面文字列はちょうど {} 行でなければならない",
            Row::NUM
        );

        let mut bcs = ColArray::<BitCol>::default();

        for (row, line) in itertools::zip_eq(Row::all().rev(), lines) {
            let chars: Vec<_> = line.chars().collect();
            ensure!(
                chars.len() == Col::NUM,
                "盤面の行 {row} がちょうど {} 文字でない",
                Col::NUM
            );

            for (col, ch) in itertools::zip_eq(Col::all(), chars) {
                let sq = Square::new(col, row);
                let piece = match ch {
                    Self::CHAR_BLANK => None,
                    '1'..='5' => Some(Piece::from_inner(ch.to_digit(10).unwrap() as u8).unwrap()),
                    _ => bail!("盤面 {sq} の文字が無効: {ch}",),
                };
                let value = piece.map_or(0, Piece::to_inner);
                bcs[col].set(row, value);
            }
        }

        let width_remain = bcs
            .as_array()
            .iter()
            .copied()
            .position(BitCol::is_zero)
            .unwrap_or(Col::NUM);
        ensure!(
            bcs.as_array()[width_remain..]
                .iter()
                .copied()
                .all(BitCol::is_zero),
            "盤面が左詰めになっていない"
        );
        let width_remain = width_remain as u32;

        Ok(Self::new(bcs, width_remain))
    }
}

impl std::fmt::Display for Board {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row in Row::all().rev() {
            for col in Col::all() {
                let sq = Square::new(col, row);
                let piece = self.get(sq);
                let ch = piece.map_or(Self::CHAR_BLANK, |piece| {
                    char::from(b'0' + piece.to_inner())
                });
                f.write_char(ch)?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

/// 盤面のマスの集合を表すマスク。
///
/// `BitCol` を `Col::NUM` 個持っている。
///
/// `BitCol` の値は、マスが集合に含まれるなら `0b001`, さもなくば `0b000` となる。
#[derive(Clone, Eq, PartialEq)]
pub struct MaskBoard {
    bcs: ColArray<BitCol>,

    /// 空でない列たちを表すマスク。
    col_mask: u32,
}

impl MaskBoard {
    const CHAR_FALSE: char = '.';
    const CHAR_TRUE: char = '*';

    /// `MaskBoard` を生成する。デバッグモードでは不変条件のチェックも行う。
    fn new(bcs: ColArray<BitCol>, col_mask: u32) -> Self {
        unsafe { assert_unchecked!((col_mask & !((1 << Col::NUM) - 1)) == 0) }

        debug_assert!(
            bcs.as_array()
                .iter()
                .copied()
                .all(|bc| bc.iter().all(|value| value <= 1)),
            "MaskBoard のマスの値は 0 または 1 でなければならない"
        );

        debug_assert!(
            Col::all().all(|col| {
                let cond_bc = !bcs[col].is_zero();
                let cond_mask = (col_mask & (1 << col.to_index())) != 0;
                cond_bc == cond_mask
            }),
            "MaskBoard: bcs と col_mask が矛盾している"
        );

        Self { bcs, col_mask }
    }

    /// 空集合を表すマスクを返す。
    pub fn empty() -> Self {
        Self::new(ColArray::default(), 0)
    }

    /// 指定したマスのみを含むマスクを返す。
    pub fn single(sq: Square) -> Self {
        let mut bcs = ColArray::<BitCol>::default();
        bcs[sq.col()].set(sq.row(), 0b001);

        Self::new(bcs, 1 << sq.col().to_index())
    }

    /// 指定したマスが集合に含まれるかどうかを返す。
    pub fn test(&self, sq: Square) -> bool {
        self.bcs[sq.col()].get(sq.row()) != 0
    }

    /// 指定したマスが集合に含まれるかどうかを設定する。
    pub fn set(&mut self, sq: Square, value: bool) {
        let bc = &mut self.bcs[sq.col()];
        let value = if value { 0b001 } else { 0b000 };

        bc.set(sq.row(), value);

        if bc.is_zero() {
            self.col_mask &= !(1 << sq.col().to_index());
        } else {
            self.col_mask |= 1 << sq.col().to_index();
        }
    }

    /// 空集合かどうかを返す。
    pub fn is_empty(&self) -> bool {
        self.col_mask == 0
    }

    /// ちょうど 1 つのマスを含むかどうかを返す。
    pub fn is_single(&self) -> bool {
        if !self.col_mask.is_power_of_two() {
            return false;
        }

        let bc = self.bcs[unsafe { self.least_nonempty_col_unchecked() }];
        bc.inner().is_power_of_two()
    }

    /// 空でない列数を返す。
    pub fn nonempty_col_count(&self) -> u32 {
        self.col_mask.count_ones()
    }

    /// 含まれるマス数を返す。
    pub fn square_count(&self) -> u32 {
        self.nonempty_cols()
            .map(|col| self.bcs[col].0.count_ones())
            .sum()
    }

    /// 空でない最小の列を返す。
    pub fn least_nonempty_col(&self) -> Option<Col> {
        (!self.is_empty()).then(|| unsafe { self.least_nonempty_col_unchecked() })
    }

    /// 空でない最小の列を返す。
    ///
    /// # Safety
    ///
    /// `self` は空であってはならない。
    pub unsafe fn least_nonempty_col_unchecked(&self) -> Col {
        let col = 1 + self.col_mask.trailing_zeros() as u8;

        unsafe { Col::from_inner_unchecked(col) }
    }

    /// 空でない列を昇順で列挙する。
    pub fn nonempty_cols(
        &self,
    ) -> impl ExactSizeIterator<Item = Col> + std::iter::FusedIterator + Clone {
        bitop::u32_one_indexs(self.col_mask).map(|i| {
            let col = 1 + i as u8;
            unsafe { Col::from_inner_unchecked(col) }
        })
    }

    /// 含まれる最小のマスを返す。
    pub fn least_square(&self) -> Option<Square> {
        (!self.is_empty()).then(|| unsafe { self.least_square_unchecked() })
    }

    /// 含まれる最小のマスを返す。
    ///
    /// # Safety
    ///
    /// `self` は空であってはならない。
    pub unsafe fn least_square_unchecked(&self) -> Square {
        assert_unchecked!(!self.is_empty());

        let col = self.least_nonempty_col_unchecked();

        let row = 1 + (self.bcs[col].0.trailing_zeros() / 3) as u8;
        let row = Row::from_inner_unchecked(row);

        Square::new(col, row)
    }

    /// 含まれるマスを昇順で列挙する。
    pub fn squares(&self) -> impl std::iter::FusedIterator<Item = Square> + Clone + '_ {
        self.nonempty_cols().flat_map(|col| {
            let bc = self.bcs[col];
            bitop::u64_one_indexs(bc.inner()).map(move |i| {
                let row = 1 + (i / 3) as u8;
                let row = unsafe { Row::from_inner_unchecked(row) };
                Square::new(col, row)
            })
        })
    }

    /// 差集合 `self` - `rhs` を返す。
    pub fn subtract(&self, rhs: &Self) -> Self {
        let mut res = self.clone();
        res.subtract_assign(rhs);
        res
    }

    /// `self` を差集合 `self` - `rhs` とする。
    pub fn subtract_assign(&mut self, rhs: &Self) {
        for col in rhs.nonempty_cols() {
            let bc = &mut self.bcs[col];
            bc.0 &= !rhs.bcs[col].0;
            if bc.is_zero() {
                self.col_mask &= !(1 << col.to_index());
            }
        }
    }

    /// 連結成分を列挙する (4 近傍)。
    pub fn components(&self) -> impl std::iter::FusedIterator<Item = Self> + Clone {
        let mut remain = self.clone();

        std::iter::from_fn(move || {
            if remain.is_empty() {
                return None;
            }

            let comp = remain.flood_fill_impl(remain.blsi());
            remain.subtract_assign(&comp);

            Some(comp)
        })
        .fuse()
    }

    /// `self` に対して `sq` を始点として flood fill を行った結果を返す。
    ///
    /// `self` は `sq` を含んでいなければならない。
    pub fn flood_fill(&self, sq: Square) -> Self {
        unsafe { assert_unchecked!(self.test(sq)) }

        self.flood_fill_impl(Self::single(sq))
    }

    fn flood_fill_impl(&self, seed: Self) -> Self {
        unsafe { assert_unchecked!(seed.is_single()) }
        unsafe { assert_unchecked!(self.test(seed.least_square_unchecked())) }

        fn col(col: u8) -> Col {
            unsafe { assert_unchecked!(matches!(col, Col::MIN_VALUE..=Col::MAX_VALUE)) }
            unsafe { Col::from_inner_unchecked(col) }
        }

        macro_rules! update {
            ($lhs:expr, $rhs:expr) => {{
                if $lhs != $rhs {
                    $lhs = $rhs;
                    true
                } else {
                    false
                }
            }};
        }

        let MaskBoard {
            mut bcs,
            mut col_mask,
        } = seed;
        let mut c_min = 1 + col_mask.trailing_zeros() as u8;
        let mut c_max = c_min;

        loop {
            let mut updated = false;

            // 上下に伸ばす。
            for c in c_min..=c_max {
                let bc = bcs[col(c)].0;
                let bc = (bc | (bc << 3) | (bc >> 3)) & self.bcs[col(c)].0;
                updated |= update!(bcs[col(c)], BitCol::new(bc));
            }
            // 左に伸ばす (左端を除く)。
            for c in c_min + 1..=c_max {
                let bc = (bcs[col(c - 1)] | bcs[col(c)]) & self.bcs[col(c - 1)];
                updated |= update!(bcs[col(c - 1)], bc);
            }
            // 右に伸ばす (右端を除く)。
            for c in c_min + 1..=c_max {
                let bc = (bcs[col(c - 1)] | bcs[col(c)]) & self.bcs[col(c)];
                updated |= update!(bcs[col(c)], bc);
            }
            // 左端を左に伸ばす。
            if c_min != Col::MIN_VALUE {
                let bc = bcs[col(c_min)] & self.bcs[col(c_min - 1)];
                if !bc.is_zero() {
                    bcs[col(c_min - 1)] = bc;
                    col_mask |= 1 << (c_min - 1 - 1);
                    c_min -= 1;
                    updated = true;
                }
            }
            // 右端を右に伸ばす。
            if c_max != Col::MAX_VALUE {
                let bc = bcs[col(c_max)] & self.bcs[col(c_max + 1)];
                if !bc.is_zero() {
                    bcs[col(c_max + 1)] = bc;
                    col_mask |= 1 << (c_max + 1 - 1);
                    c_max += 1;
                    updated = true;
                }
            }

            if !updated {
                break Self::new(bcs, col_mask);
            }
        }
    }

    /// `self` の最下位マスのみが含まれるマスクを返す。
    ///
    /// `self` は空であってはならない。
    fn blsi(&self) -> Self {
        unsafe { assert_unchecked!(!self.is_empty()) }

        let mut bcs = ColArray::<BitCol>::default();

        let col = unsafe { self.least_nonempty_col_unchecked() };
        bcs[col].0 = bitop::u64_blsi(self.bcs[col].0);

        Self::new(bcs, 1 << col.to_index())
    }
}

impl std::fmt::Debug for MaskBoard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MaskBoard")
            .field("bcs", &self.bcs)
            .field("col_mask", &ColMaskDebug(self.col_mask))
            .finish()
    }
}

struct ColMaskDebug(u32);

impl std::fmt::Debug for ColMaskDebug {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0b{:0width$b}", self.0, width = Col::NUM)
    }
}

impl std::str::FromStr for MaskBoard {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lines: Vec<_> = s.lines().collect();
        ensure!(
            lines.len() == Row::NUM,
            "盤面マスク文字列はちょうど {} 行でなければならない",
            Row::NUM
        );

        let mut this = Self::empty();

        for (row, line) in itertools::zip_eq(Row::all().rev(), lines) {
            let chars: Vec<_> = line.chars().collect();
            ensure!(
                chars.len() == Col::NUM,
                "盤面の行 {row} がちょうど {} 文字でない",
                Col::NUM
            );

            for (col, ch) in itertools::zip_eq(Col::all(), chars) {
                let sq = Square::new(col, row);
                let value = match ch {
                    Self::CHAR_FALSE => false,
                    Self::CHAR_TRUE => true,
                    _ => bail!("盤面マスク {sq} の文字が無効: {ch}",),
                };
                this.set(sq, value);
            }
        }

        Ok(this)
    }
}

impl std::fmt::Display for MaskBoard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row in Row::all().rev() {
            for col in Col::all() {
                let sq = Square::new(col, row);
                let ch = if self.test(sq) {
                    Self::CHAR_TRUE
                } else {
                    Self::CHAR_FALSE
                };
                f.write_char(ch)?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use itertools::assert_equal;

    use crate::square::*;

    use super::*;

    fn sq_new(col: Col, row: Row) -> Square {
        Square::new(col, row)
    }

    fn parse_board(s: impl AsRef<str>) -> Board {
        s.as_ref().parse().unwrap()
    }

    fn parse_mask_board(s: impl AsRef<str>) -> MaskBoard {
        s.as_ref().parse().unwrap()
    }

    #[test]
    fn test_bit_col() {
        let mut bc = BitCol::new(0b011_010_001_000_111_110_101_100_011_010_001_000);
        bc.set(ROW_2, 0b101);

        assert_eq!(bc.get(ROW_1), 0b000);
        assert_eq!(bc.get(ROW_2), 0b101);
        assert_eq!(bc.get(ROW_3), 0b010);
        assert_eq!(bc.get(ROW_4), 0b011);
        assert_eq!(bc.get(ROW_5), 0b100);
        assert_eq!(bc.get(ROW_6), 0b101);
        assert_eq!(bc.get(ROW_7), 0b110);
        assert_eq!(bc.get(ROW_8), 0b111);
        assert_eq!(bc.get(ROW_9), 0b000);
        assert_eq!(bc.get(ROW_10), 0b001);
        assert_eq!(bc.get(ROW_11), 0b010);
        assert_eq!(bc.get(ROW_12), 0b011);

        assert_equal(
            bc.enumerate(),
            [
                (ROW_1, 0b000),
                (ROW_2, 0b101),
                (ROW_3, 0b010),
                (ROW_4, 0b011),
                (ROW_5, 0b100),
                (ROW_6, 0b101),
                (ROW_7, 0b110),
                (ROW_8, 0b111),
                (ROW_9, 0b000),
                (ROW_10, 0b001),
                (ROW_11, 0b010),
                (ROW_12, 0b011),
            ],
        );
    }

    #[test]
    fn test_board_io() {
        let cases = [
            indoc! {"
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
            "},
            indoc! {"
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                .1.............
                121............
                1213...........
                1213...........
            "},
            indoc! {"
                234512345123451
                123451234512345
                512345123451234
                451234512345123
                345123451234512
                234512345123451
                123451234512345
                512345123451234
                451234512345123
                345123451234512
                234512345123451
                123451234512345
            "},
        ];

        for s in cases {
            let board = parse_board(s);
            assert_eq!(board.to_string(), s);
        }
    }

    #[test]
    fn test_board_piece_count() {
        for piece in Piece::all() {
            assert_eq!(Board::empty().piece_count(piece), 0);
        }
        assert_eq!(Board::empty().piece_count_total(), 0);

        let cases = [
            (
                indoc! {"
                    ...............
                    ...............
                    ...............
                    ...............
                    ...............
                    ...............
                    ...............
                    ...............
                    ...............
                    ...............
                    ...............
                    12345..........
                "},
                [1, 1, 1, 1, 1],
            ),
            (
                indoc! {"
                    .......4.......
                    .......4.......
                    .......4.......
                    .......4.......
                    .......4.......
                    .......4.4.....
                    .......4.2.....
                    .......4.2.....
                    .1...5.423.....
                    .1.3.5.423....1
                    1213.5.555....2
                    121345551511122
                "},
                [11, 9, 5, 12, 10],
            ),
        ];

        for (board, counts) in cases {
            let board = parse_board(board);
            for piece in Piece::all() {
                assert_eq!(board.piece_count(piece), counts[piece.to_index()]);
            }
            assert_eq!(board.piece_count_total(), counts.into_iter().sum());
        }
    }

    #[test]
    fn test_board_has_action() {
        assert!(!Board::empty().has_action());

        let falses = [
            indoc! {"
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...543.........
                ..14213........
                1232121........
            "},
            indoc! {"
                234512345123451
                123451234512345
                512345123451234
                451234512345123
                345123451234512
                234512345123451
                123451234512345
                512345123451234
                451234512345123
                345123451234512
                234512345123451
                123451234512345
            "},
        ];

        let trues = [
            indoc! {"
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                1..............
                1.5............
                234............
            "},
            indoc! {"
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                .34............
                2251...........
            "},
            indoc! {"
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                123451234512355
            "},
            indoc! {"
                ..............5
                ..............5
                ..............4
                ..............3
                ..............2
                ..............1
                ..............5
                ..............4
                ..............3
                ..............2
                ..............1
                123451234512345
            "},
            indoc! {"
                1...........232
                1...........222
                1...........232
                1...........232
                1...........222
                1...........352
                1...........222
                155.........211
                111.4..55...222
                12144..35...132
                12133.555...122
                121355545511144
            "},
        ];

        for board in falses {
            let board = parse_board(board);
            assert!(!board.has_action());
        }
        for board in trues {
            let board = parse_board(board);
            assert!(board.has_action());
        }
    }

    #[test]
    fn test_board_erase() {
        let cases = [
            (
                indoc! {"
                    1...........232
                    1...........222
                    1...........232
                    1...........232
                    1...........222
                    1...........352
                    1...........222
                    155.........211
                    111.4..55...222
                    12144..35...132
                    12133.555...122
                    121355545511144
                "},
                indoc! {"
                    *..............
                    *..............
                    *..............
                    *..............
                    *..............
                    *..............
                    *..............
                    *..............
                    ***............
                    *.*............
                    *.*............
                    *.*............
                "},
                indoc! {"
                    ...........232.
                    ...........222.
                    ...........232.
                    ...........232.
                    ...........222.
                    ...........352.
                    ...........222.
                    ...........211.
                    5..4..55...222.
                    2.44..35...132.
                    2.33.555...122.
                    25355545511144.
                "},
            ),
            (
                indoc! {"
                    1...........232
                    1...........222
                    1...........232
                    1...........232
                    1...........222
                    1...........352
                    1...........222
                    155.........211
                    111.4..55...222
                    12144..35...132
                    12133.555...122
                    121355545511144
                "},
                indoc! {"
                    ...............
                    ...............
                    ...............
                    ...............
                    ...............
                    ...............
                    ...............
                    .**............
                    ...............
                    ...............
                    ...............
                    ...............
                "},
                indoc! {"
                    1...........232
                    1...........222
                    1...........232
                    1...........232
                    1...........222
                    1...........352
                    1...........222
                    1...........211
                    111.4..55...222
                    12144..35...132
                    12133.555...122
                    121355545511144
                "},
            ),
            (
                indoc! {"
                    1...........232
                    1...........222
                    1...........232
                    1...........232
                    1...........222
                    1...........352
                    1...........222
                    155.........211
                    111.4..55...222
                    12144..35...132
                    12133.555...122
                    121355545511144
                "},
                indoc! {"
                    ...............
                    ...............
                    ...............
                    ...............
                    ...............
                    ...............
                    ...............
                    ...............
                    ...............
                    ...............
                    ...**..........
                    ...*...........
                "},
                indoc! {"
                    1...........232
                    1...........222
                    1...........232
                    1...........232
                    1...........222
                    1...........352
                    1...........222
                    155.........211
                    111....55...222
                    121.4..35...132
                    121.4.555...122
                    121455545511144
                "},
            ),
            (
                indoc! {"
                    1...........232
                    1...........222
                    1...........232
                    1...........232
                    1...........222
                    1...........352
                    1...........222
                    155.........211
                    111.4..55...222
                    12144..35...132
                    12133.555...122
                    121355545511144
                "},
                indoc! {"
                    ...............
                    ...............
                    ...............
                    ...............
                    ...............
                    ...............
                    ...............
                    ...............
                    .......**......
                    ........*......
                    ......***......
                    ....***.**.....
                "},
                indoc! {"
                    1.......232....
                    1.......222....
                    1.......232....
                    1.......232....
                    1.......222....
                    1.......352....
                    1.......222....
                    155.....211....
                    111.....222....
                    12144...132....
                    121343..122....
                    12133411144....
                "},
            ),
            (
                indoc! {"
                    1...........232
                    1...........222
                    1...........232
                    1...........232
                    1...........222
                    1...........352
                    1...........222
                    155.........211
                    111.4..55...222
                    12144..35...132
                    12133.555...122
                    121355545511144
                "},
                indoc! {"
                    ...............
                    ...............
                    ...............
                    ...............
                    ...............
                    ...............
                    ...............
                    ...............
                    ...............
                    ............*..
                    ............*..
                    ..........***..
                "},
                indoc! {"
                    1..........32..
                    1..........22..
                    1..........32..
                    1.........232..
                    1.........222..
                    1.........252..
                    1.........222..
                    155.......211..
                    111.4..55.322..
                    12144..35.232..
                    12133.555.222..
                    1213555455244..
                "},
            ),
            (
                indoc! {"
                    1...........232
                    1...........222
                    1...........232
                    1...........232
                    1...........222
                    1...........352
                    1...........222
                    155.........211
                    111.4..55...222
                    12144..35...132
                    12133.555...122
                    121355545511144
                "},
                indoc! {"
                    ............*.*
                    ............***
                    ............*.*
                    ............*.*
                    ............***
                    ..............*
                    ............***
                    ............*..
                    ............***
                    ..............*
                    .............**
                    ...............
                "},
                indoc! {"
                    1..............
                    1..............
                    1..............
                    1..............
                    1..............
                    1............3.
                    1............3.
                    155..........3.
                    111.4..55...35.
                    12144..35...11.
                    12133.555...131
                    121355545511144
                "},
            ),
        ];

        for (before, mb, after) in cases {
            let before = parse_board(before);
            let mb = parse_mask_board(mb);
            let after = parse_board(after);

            assert_eq!(before.erase(&mb), after);
        }
    }

    #[test]
    fn test_board_xor_mask() {
        assert_eq!(Board::empty().xor_mask(&Board::empty()), MaskBoard::empty());

        {
            let board = parse_board(indoc! {"
                234512345123451
                123451234512345
                512345123451234
                451234512345123
                345123451234512
                234512345123451
                123451234512345
                512345123451234
                451234512345123
                345123451234512
                234512345123451
                123451234512345
            "});
            assert_eq!(board.xor_mask(&board), MaskBoard::empty());
        }

        let cases = [
            (
                indoc! {"
                    1...........232
                    1...........222
                    1...........232
                    1...........232
                    1...........222
                    1...........352
                    1...........222
                    155.........211
                    111.4..55...222
                    12144..35...132
                    12133.555...122
                    121355545511144
                "},
                indoc! {"
                    ...........232.
                    ...........222.
                    ...........232.
                    ...........232.
                    ...........222.
                    ...........352.
                    ...........222.
                    ...........211.
                    5..4..55...222.
                    2.44..35...132.
                    2.33.555...122.
                    25355545511144.
                "},
                indoc! {"
                    *..........****
                    *..........*..*
                    *..........****
                    *..........****
                    *..........*..*
                    *..........****
                    *..........*..*
                    ***........**.*
                    *****.*.*..*..*
                    ***.*.***..****
                    ***.**..*..**.*
                    ****..**.*..*.*
                "},
            ),
            (
                indoc! {"
                    1...........232
                    1...........222
                    1...........232
                    1...........232
                    1...........222
                    1...........352
                    1...........222
                    155.........211
                    111.4..55...222
                    12144..35...132
                    12133.555...122
                    121355545511144
                "},
                indoc! {"
                    1...........232
                    1...........222
                    1...........232
                    1...........232
                    1...........222
                    1...........352
                    1...........222
                    1...........211
                    111.4..55...222
                    12144..35...132
                    12133.555...122
                    121355545511144
                "},
                indoc! {"
                    ...............
                    ...............
                    ...............
                    ...............
                    ...............
                    ...............
                    ...............
                    .**............
                    ...............
                    ...............
                    ...............
                    ...............
                "},
            ),
            (
                indoc! {"
                    1...........232
                    1...........222
                    1...........232
                    1...........232
                    1...........222
                    1...........352
                    1...........222
                    155.........211
                    111.4..55...222
                    12144..35...132
                    12133.555...122
                    121355545511144
                "},
                indoc! {"
                    1..............
                    1..............
                    1..............
                    1..............
                    1..............
                    1............3.
                    1............3.
                    155..........3.
                    111.4..55...35.
                    12144..35...11.
                    12133.555...131
                    121355545511144
                "},
                indoc! {"
                    ............***
                    ............***
                    ............***
                    ............***
                    ............***
                    ............***
                    ............***
                    ............***
                    ............***
                    .............**
                    .............**
                    ...............
                "},
            ),
        ];

        for (before, after, mb) in cases {
            let before = parse_board(before);
            let after = parse_board(after);
            let mb = parse_mask_board(mb);

            assert_eq!(before.xor_mask(&after), mb);
            assert_eq!(after.xor_mask(&before), mb);
        }
    }

    #[test]
    fn test_mask_board_io() {
        let cases = [
            indoc! {"
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
            "},
            indoc! {"
                *.............*
                .*.............
                ..*......***...
                ...*.....*.*...
                ....*....*....*
                ...*.*......*.*
                .*....*.....*..
                .......*...*...
                ...**.....*....
                *.*......*....*
                *.*.*...*....**
                *.*..*..*..**.*
            "},
        ];

        for s in cases {
            let mb = parse_mask_board(s);
            assert_eq!(mb.to_string(), s);
        }
    }

    #[test]
    fn test_mask_board_squares() {
        assert_eq!(MaskBoard::empty().squares().next(), None);

        let cases = [
            (
                indoc! {"
                    *.............*
                    ...............
                    ...............
                    ...............
                    ...............
                    ...............
                    ...............
                    ...............
                    ...............
                    ...............
                    ...............
                    *.............*
                "},
                vec![
                    sq_new(COL_1, ROW_1),
                    sq_new(COL_1, ROW_12),
                    sq_new(COL_15, ROW_1),
                    sq_new(COL_15, ROW_12),
                ],
            ),
            (
                indoc! {"
                    .......*.......
                    ......*.*......
                    .....*...*.....
                    ....*.....*....
                    ...*.......*...
                    ..*.........*..
                    .*...........*.
                    *.............*
                    ...............
                    ...............
                    ...............
                    ...............
                "},
                vec![
                    sq_new(COL_1, ROW_5),
                    sq_new(COL_2, ROW_6),
                    sq_new(COL_3, ROW_7),
                    sq_new(COL_4, ROW_8),
                    sq_new(COL_5, ROW_9),
                    sq_new(COL_6, ROW_10),
                    sq_new(COL_7, ROW_11),
                    sq_new(COL_8, ROW_12),
                    sq_new(COL_9, ROW_11),
                    sq_new(COL_10, ROW_10),
                    sq_new(COL_11, ROW_9),
                    sq_new(COL_12, ROW_8),
                    sq_new(COL_13, ROW_7),
                    sq_new(COL_14, ROW_6),
                    sq_new(COL_15, ROW_5),
                ],
            ),
        ];

        for (mb, sqs) in cases {
            let mb = parse_mask_board(mb);
            assert_equal(mb.squares(), sqs);
        }
    }

    #[test]
    fn test_mask_board_components() {
        assert_eq!(MaskBoard::empty().components().next(), None);

        let mb = parse_mask_board(indoc! {"
            ******........*
            .....*.........
            .....***.......
            ..****.*.......
            ...*...*.......
            ...*..**.......
            ...*......***..
            ...*.....*..*..
            ...*........*..
            ***.*........*.
            *..***.......*.
            *...*........**
        "});
        let expect = [
            indoc! {"
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ***............
                *..............
                *..............
            "},
            indoc! {"
                ******.........
                .....*.........
                .....***.......
                ..****.*.......
                ...*...*.......
                ...*..**.......
                ...*...........
                ...*...........
                ...*...........
                ...............
                ...............
                ...............
            "},
            indoc! {"
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ....*..........
                ...***.........
                ....*..........
            "},
            indoc! {"
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                .........*.....
                ...............
                ...............
                ...............
                ...............
            "},
            indoc! {"
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ..........***..
                ............*..
                ............*..
                ...............
                ...............
                ...............
            "},
            indoc! {"
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                .............*.
                .............*.
                .............**
            "},
            indoc! {"
                ..............*
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
                ...............
            "},
        ];

        let expect = expect.map(parse_mask_board);
        assert_equal(mb.components(), expect);
    }
}
