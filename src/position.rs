//! 局面関連。

use crate::action::Action;
use crate::board::Board;
use crate::piece::{Piece, PieceArray};
use crate::score::{calc_score_erase, Score, SCORE_PERFECT};
use crate::square::Square;
use crate::zobrist::ZOBRIST_TABLE;

/// 局面。
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Position {
    board: Board,
    key: u64,
    piece_counts: PieceArray<u8>,
}

impl Position {
    /// 初期盤面を指定して局面を作る。
    pub fn new(board: Board) -> Self {
        let key = Square::all()
            .map(|sq| {
                board
                    .get(sq)
                    .map_or(0, |piece| ZOBRIST_TABLE.board(piece, sq))
            })
            .reduce(std::ops::BitXor::bitxor)
            .unwrap();

        let piece_counts = PieceArray::from_fn(|piece| board.piece_count(piece) as u8);

        Self {
            board,
            key,
            piece_counts,
        }
    }

    /// 盤面を返す。
    pub fn board(&self) -> &Board {
        &self.board
    }

    /// ハッシュ値を返す。
    pub fn key(&self) -> u64 {
        self.key
    }

    /// 指定した駒種の数を返す。
    pub fn piece_count(&self, piece: Piece) -> u8 {
        self.piece_counts[piece]
    }

    /// 合法手があるかどうかを返す。
    pub fn has_action(&self) -> bool {
        self.board().has_action()
    }

    /// 合法手を列挙する。
    pub fn actions(&self) -> impl std::iter::FusedIterator<Item = Action> + Clone + '_ {
        self.board
            .piece_components()
            .filter(|(_piece, mb)| !mb.is_single())
            .map(|(piece, mb)| Action::new(piece, mb))
    }

    /// 着手を行い、結果の局面を返す。
    pub fn do_action(&self, action: &Action) -> Self {
        let board = self.board.erase(action.mask());

        let mut key = self.key;
        for sq in self.board.xor_mask(&board).squares() {
            // 着手前、sq には駒があったとは限らないことに注意(列が詰め直されるケースがあるので)。
            if let Some(piece_before) = self.board.get(sq) {
                key ^= ZOBRIST_TABLE.board(piece_before, sq);
            }
            if let Some(piece_after) = board.get(sq) {
                key ^= ZOBRIST_TABLE.board(piece_after, sq);
            }
        }

        let mut piece_counts = self.piece_counts.clone();
        piece_counts[action.piece()] -= action.square_count() as u8;

        Self {
            board,
            key,
            piece_counts,
        }
    }

    /// この局面から追加で獲得しうるスコアの上界を返す。
    /// 探索は一切行わず、粗く見積もる。
    ///
    /// 盤面が空の場合、追加でパーフェクトボーナスを獲得できるとみなす。
    ///
    /// この関数が 0 を返すならば、`self` はパーフェクトでない終了局面である。
    /// ただし逆は成り立たない (例: `121.......`)。
    pub fn gain_upper_bound(&self) -> Score {
        // 2 個以上存在する駒種全てが 1 手で全消しできると仮定して上界を求める。
        // 適宜パーフェクトボーナスも加算する。

        let mut res = 0;
        let mut perfect = true;
        for piece in Piece::all() {
            let count = self.piece_count(piece);
            match count {
                0 => {}
                1 => perfect = false,
                _ => res += calc_score_erase(u32::from(count)),
            }
        }

        if perfect {
            res += SCORE_PERFECT;
        }

        res
    }
}

impl std::hash::Hash for Position {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.key.hash(state)
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;

    use crate::hash::U64HashMap;
    use crate::square::*;

    use super::*;

    fn sq_new(col: Col, row: Row) -> Square {
        Square::new(col, row)
    }

    fn parse_board(s: impl AsRef<str>) -> Board {
        s.as_ref().parse().unwrap()
    }

    fn pos_do_action(pos: &Position, sq: Square) -> Position {
        let action = Action::from_board_square(pos.board(), sq).unwrap();
        pos.do_action(&action)
    }

    #[test]
    fn test_position() {
        assert_eq!(Position::new(Board::empty()).key(), 0);

        let pos_start = Position::new(parse_board(indoc! {"
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
        "}));
        let pos = pos_do_action(&pos_start, sq_new(COL_2, ROW_5));
        let pos = pos_do_action(&pos, sq_new(COL_1, ROW_1));
        let pos = pos_do_action(&pos, sq_new(COL_11, ROW_4));

        let pos_expect = Position::new(parse_board(indoc! {"
            ...............
            ...............
            ...............
            ...............
            ...............
            ...........3...
            ...........3...
            ...........3...
            ..4..55...35...
            244..35...11...
            233.555...131..
            2355545511144..
        "}));
        assert_eq!(pos.board(), pos_expect.board());
        assert_eq!(pos.key(), pos_expect.key());
        for piece in Piece::all() {
            assert_eq!(pos.piece_count(piece), pos_expect.piece_count(piece));
        }
    }

    #[test]
    fn test_hash() {
        let pos1 = Position::new(parse_board(indoc! {"
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
        "}));
        let pos2 = Position::new(parse_board(indoc! {"
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
        "}));
        let pos3 = Position::new(parse_board(indoc! {"
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
        "}));

        let mut map = U64HashMap::<Position, u32>::default();
        map.insert(pos1.clone(), 1);
        map.insert(pos2.clone(), 2);
        map.insert(pos3.clone(), 3);

        assert_eq!(map.get(&pos1), Some(&1));
        assert_eq!(map.get(&pos2), Some(&2));
        assert_eq!(map.get(&pos3), Some(&3));
    }
}
