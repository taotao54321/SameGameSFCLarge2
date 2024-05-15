//! ゲーム内乱数関連。

use anyhow::{anyhow, ensure, Context as _};
use arrayvec::ArrayVec;

use crate::board::Board;
use crate::bounded::impl_bounded_uint;
use crate::hint::assert_unchecked;
use crate::piece::Piece;
use crate::square::{Col, ColArray, RowArray, Square};

/// 全ての盤面生成パラメータについて (生成パラメータ, 盤面, ゲーム内に出現しうるか, 生成後の乱数生成器) を列挙する。
pub fn enumerate_all_board(
) -> impl std::iter::FusedIterator<Item = (RandomBoardParam, Board, bool, GameRng)> {
    RandomBoardParam::all().map(|param| {
        let (board, legal, rng_after) = param.gen_board();
        (param, board, legal, rng_after)
    })
}

/// ゲーム内に現れうる全ての盤面について (生成パラメータ, 盤面, 生成後の乱数生成器) を列挙する。
pub fn enumerate_all_legal_board(
) -> impl std::iter::FusedIterator<Item = (RandomBoardParam, Board, GameRng)> {
    enumerate_all_board()
        .filter_map(|(param, board, legal, rng_after)| legal.then_some((param, board, rng_after)))
}

/// 盤面生成にわずかな影響を与えるゲーム内エントロピー。値域は `0..=4`。
///
/// メインループカウンタ `$7F0046` から生成される (式は `(5 * counter) >> 8`)。
///
/// 値とメインループカウンタの対応は以下の通り:
///
/// | 値   | カウンタ範囲 |
/// | --   | --           |
/// | `0`  | `  0..= 51`  |
/// | `1`  | ` 52..=102`  |
/// | `2`  | `103..=153`  |
/// | `3`  | `154..=204`  |
/// | `4`  | `205..=255`  |
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GameEntropy(u8);

impl_bounded_uint!(GameEntropy, u8, 4);

impl std::str::FromStr for GameEntropy {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let entropy: u8 = s
            .parse()
            .with_context(|| format!("GameEntropy のパースに失敗: '{s}'"))?;

        GameEntropy::from_inner(entropy).ok_or_else(|| anyhow!("GameEntropy の値が無効: {entropy}"))
    }
}

impl std::fmt::Display for GameEntropy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// ランダムな盤面を生成するためのパラメータ。
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RandomBoardParam {
    /// 乱数生成器の内部状態。
    pub rng_state: u16,
    /// NMI カウンタ。
    pub nmi_counter: u8,
    /// 盤面生成中の NMI 発生タイミング (駒をこの個数だけ生成した直後に NMI カウンタをインクリメントする)。
    /// ゲーム内では通常 40 になるようだ (盤面再生成時はまた別)。
    pub nmi_timing: usize,
    /// ゲーム内エントロピー。
    pub entropy: GameEntropy,
}

impl RandomBoardParam {
    /// このパラメータで盤面を生成する。
    /// (盤面, ゲーム内に出現しうるか, 生成後の乱数生成器) を返す。
    pub fn gen_board(&self) -> (Board, bool, GameRng) {
        let mut rng = GameRng::new(self.rng_state);
        let (board, legal) = rng.gen_board(self.nmi_counter, self.nmi_timing, self.entropy);

        (board, legal, rng)
    }

    /// このパラメータで有効な盤面を生成する。
    /// 盤面がゲーム内に現れない場合、`None` を返す。
    pub fn gen_legal_board(&self) -> Option<(Board, GameRng)> {
        let (board, legal, rng_after) = self.gen_board();

        legal.then_some((board, rng_after))
    }

    /// 全パラメータを昇順で列挙する。
    ///
    /// 乱数生成器の内部状態の bit15 は実質無意味なので、範囲は `0..=0x7FFF` としている。
    /// NMI 発生タイミングは 40 固定としている。
    pub fn all() -> impl std::iter::FusedIterator<Item = Self> + Clone {
        itertools::iproduct!(0..=0x7FFF, 0..=u8::MAX, 40..=40, GameEntropy::all())
            .map(|(rng_state, nmi_counter, nmi_timing, entropy)| Self {
                rng_state,
                nmi_counter,
                nmi_timing,
                entropy,
            })
            .fuse()
    }
}

impl std::str::FromStr for RandomBoardParam {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let fields: Vec<_> = s.split(',').collect();
        ensure!(
            fields.len() == 4,
            "RandomBoardParam 文字列はカンマ区切りの 4 フィールドでなければならない"
        );

        let rng_state: u16 = parse_int::parse(fields[0])
            .with_context(|| format!("rng_state のパースに失敗: '{}'", fields[0]))?;
        let nmi_counter: u8 = parse_int::parse(fields[1])
            .with_context(|| format!("nmi_counter のパースに失敗: '{}'", fields[1]))?;
        let nmi_timing: usize = fields[2]
            .parse()
            .with_context(|| format!("nmi_timing のパースに失敗: '{}'", fields[2]))?;
        let entropy: GameEntropy = fields[3]
            .parse()
            .with_context(|| format!("entropy のパースに失敗: '{}'", fields[3]))?;

        Ok(Self {
            rng_state,
            nmi_counter,
            nmi_timing,
            entropy,
        })
    }
}

impl std::fmt::Display for RandomBoardParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "0x{:04X},0x{:02X},{},{}",
            self.rng_state, self.nmi_counter, self.nmi_timing, self.entropy
        )
    }
}

/// ゲーム内の乱数生成器。
///
/// 16bit シフトレジスタだが、NMI カウンタ `$7F0F52` およびゲーム内エントロピーの影響を受ける。
///
/// NOTE: 内部状態の最上位ビット (bit15) は意味を持たない。
/// `gen()` の更新式を見ての通り、bit15 は単に捨てられている。
#[repr(transparent)]
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct GameRng(u16);

impl GameRng {
    /// 内部状態を与えて乱数生成器を作る。
    pub const fn new(state: u16) -> Self {
        Self(state)
    }

    /// 内部状態を返す。
    pub const fn state(self) -> u16 {
        self.0
    }

    /// 内部状態を更新し、`0..=0xFF` の乱数を返す。
    /// NMI カウンタの影響を受ける。
    pub fn gen(&mut self, nmi_counter: u8) -> u8 {
        let bit = ((self.0 >> 14) ^ self.0) & 1;

        self.0 = self.0 ^ ((self.0 << 8) | u16::from(nmi_counter));
        self.0 = (self.0 << 1) | bit;

        (self.0 ^ (self.0 >> 8)) as u8
    }

    /// ランダムな駒を生成する。
    /// NMI カウンタおよびゲーム内エントロピーの影響を受ける。
    pub fn gen_piece(&mut self, nmi_counter: u8, entropy: GameEntropy) -> Piece {
        // 0..5 の乱数を発生。
        let r = self.gen(nmi_counter);
        let r = ((5 * u32::from(r) + u32::from(entropy.to_inner())) >> 8) as u8;

        unsafe { Piece::from_inner_unchecked(1 + r) }
    }

    /// ランダムな盤面を生成する。
    /// NMI カウンタ、本体 CPU サイクル、ゲーム内エントロピーの影響を受ける。
    ///
    /// 戻り値の `bool` は、この盤面がゲーム中に出現しうるなら `true`,
    /// 再生成判定に引っかかって出現しえないなら `false` となる。
    ///
    /// `nmi_timing` は、駒を何個生成した後に NMI カウンタをインクリメントするかのパラメータ。
    /// (ゲーム内では盤面生成中に NMI が発生して NMI カウンタがインクリメントされる。
    /// 通常は駒が 40 個生成された直後に NMI が発生するようだが、
    /// 盤面再生成時はタイミングが異なる (46 個生成直後の NMI 発生を確認している)。
    pub fn gen_board(
        &mut self,
        nmi_counter: u8,
        nmi_timing: usize,
        entropy: GameEntropy,
    ) -> (Board, bool) {
        unsafe { assert_unchecked!(nmi_timing <= Square::NUM) }

        // row-major (下から上の順)
        let mut pieces = ArrayVec::<Piece, { Square::NUM }>::new();
        pieces.extend(
            std::iter::repeat_with(|| self.gen_piece(nmi_counter, entropy)).take(nmi_timing),
        );
        pieces.extend(
            std::iter::repeat_with(|| self.gen_piece(nmi_counter.wrapping_add(1), entropy))
                .take(Square::NUM - nmi_timing),
        );

        let arrays = ColArray::from_fn(|col| {
            RowArray::from_fn(|row| pieces[Col::NUM * row.to_index() + col.to_index()])
        });
        let board = Board::from_piece_arrays(&arrays);

        // ゲーム内では同種駒の個数が (マス数) / 2 以上の場合、盤面が再生成される。
        let legal = Piece::all().all(|piece| (board.piece_count(piece) as usize) < Square::NUM / 2);

        (board, legal)
    }
}

impl std::fmt::Debug for GameRng {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GameRng(0x{:04X}", self.0)
    }
}
