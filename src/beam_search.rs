//! 最大ビーム幅 2^32 のビームサーチ。

use std::cmp::Ordering;
use std::collections::hash_map::Entry as HashMapEntry;
use std::collections::BinaryHeap;

use arrayvec::ArrayVec;
use log::info;

use crate::action::{Action, ActionHistory};
use crate::board::Board;
use crate::cmp::chmax;
use crate::hash::{u64_hashmap_with_capacity, U64HashMap};
use crate::hint::assert_unchecked;
use crate::position::Position;
use crate::score::{Score, SCORE_PERFECT};
use crate::square::Square;

type Beams = ArrayVec<Beam, { ActionHistory::CAPACITY + 1 }>; // バグってなければ +1 は不要なはずだが一応...

/// ビームサーチによるソルバー。複数の面を連続で解ける。
#[derive(Debug)]
pub struct BeamSearch {
    /// ビーム幅。
    beam_capacity: usize,

    /// 探索時の枝刈り用スコア閾値。
    /// 最終スコアがこの値を超えないノードは枝刈りする。
    /// ただし終了局面については一応解を記録する。
    prune_score_max: Score,

    best_score: Score,
    best_solution: Option<ActionHistory>,

    /// 評価値上位 k 件の子状態を保持する。
    /// これは初期化がかなり重いのでソルバー自体に持つ。
    children: Children,
}

impl BeamSearch {
    pub fn new(beam_capacity: usize, prune_score_max: Score) -> Self {
        Self {
            beam_capacity,
            prune_score_max,

            best_score: 0,
            best_solution: None,

            children: Children::with_beam_capacity(beam_capacity),
        }
    }

    /// ビーム幅を返す。
    pub fn beam_capacity(&self) -> usize {
        self.beam_capacity
    }

    /// 現時点での枝刈り用スコア閾値を返す。
    pub fn prune_score_max(&self) -> Score {
        self.prune_score_max
    }

    /// 枝刈り用スコア閾値を設定する。
    pub fn set_prune_score_max(&mut self, prune_score_max: Score) {
        self.prune_score_max = prune_score_max;
    }

    /// 枝刈り用スコア閾値を chmax する。
    pub fn chmax_prune_score_max(&mut self, score: Score) {
        chmax!(self.prune_score_max, score);
    }

    pub fn solve(&mut self, board: Board) -> Option<(Score, ActionHistory)> {
        let pos_root = Position::new(board);

        // 初期局面が終了局面のケースを考えたくないので先に例外処理。
        // どうでもいいケースなので None を返しておく。
        if !pos_root.has_action() {
            return None;
        }

        // 初期局面の追加スコア上界が prune_score_max 以下ならば直ちに解なしとわかる。
        if pos_root.gain_upper_bound() <= self.prune_score_max {
            return None;
        }

        // 必要な初期化を行う (メソッド末尾でやってもいいが、バグらせがちなので)。
        self.best_score = 0;
        self.best_solution = None;

        let mut beams = Beams::new();
        beams.push(Beam::new_root());

        // depth は現在のビームの beams 内インデックス。
        for depth in 0..ActionHistory::CAPACITY {
            let beam = &beams[depth];

            // beam 内の各状態について子状態を列挙し、
            //
            // * この状態が終了局面なら解の更新処理を行い、次の子状態へ。
            // * この状態の最終スコア上界が prune_score_max を超えないなら枝刈りし、次の子状態へ。
            // * それ以外の場合、この状態を children に追加することを試みる。
            for (idx, state) in beam.enumerate() {
                let (score, pos, history) = Self::trace_state(&beams, &pos_root, depth, state);

                for (score_child, pos_child, state_child, parent_sq) in
                    self.state_children(idx, score, &pos)
                {
                    if let Some(score_final) = position_final_score(score_child, &pos_child) {
                        if chmax!(self.best_score, score_final) {
                            let mut solution = history.clone();
                            solution.push(parent_sq);
                            info!("Found {}: {solution}", self.best_score);
                            self.best_solution.replace(solution);
                        }
                        continue;
                    }
                    if score_child + pos_child.gain_upper_bound() <= self.prune_score_max {
                        continue;
                    }
                    self.children
                        .insert(make_child_state(score_child, &pos_child, state_child));
                }
            }

            info!(
                "Depth {depth}: beam_len={} children_len={}",
                beam.len(),
                self.children.len()
            );

            // children を次のビームとする。
            // 次のビームが空なら終了。
            let beam_nxt = self.children.drain_to_beam();
            if beam_nxt.is_empty() {
                break;
            }
            beams.push(beam_nxt);
        }

        self.best_solution
            .take()
            .map(|solution| (self.best_score, solution))
    }

    /// 深さ `depth` の状態 `state` に対して経路復元を行い、(スコア, 局面, 手順) を得る。
    fn trace_state(
        beams: &Beams,
        pos_root: &Position,
        mut depth: usize,
        mut state: State,
    ) -> (Score, Position, ActionHistory) {
        let mut sqs = ArrayVec::<Square, { ActionHistory::CAPACITY }>::new();
        while let Some((parent_idx, parent_sq)) = state.parent() {
            sqs.push(parent_sq);
            depth -= 1;
            state = beams[depth].get(parent_idx);
        }

        let history: ActionHistory = sqs.into_iter().rev().collect();

        let mut score = 0;
        let mut pos = pos_root.clone();
        for &sq in &history {
            let action = unsafe { Action::from_board_square_unchecked(pos.board(), sq) };
            score += action.gain();
            pos = pos.do_action(&action);
        }

        (score, pos, history)
    }

    /// 現在のビーム内インデックス `idx` の状態の子を列挙する。
    fn state_children<'pos>(
        &self,
        idx: usize,
        score: Score,
        pos: &'pos Position,
    ) -> impl std::iter::FusedIterator<Item = (Score, Position, State, Square)> + 'pos {
        pos.actions().map(move |action| {
            let sq = action.least_square();
            let score = score + action.gain();
            let pos = pos.do_action(&action);
            let state = State::new_child(idx, sq);
            (score, pos, state, sq)
        })
    }
}

/// スコア `score` の局面 `pos` が終了局面ならばその最終スコアを、さもなくば `None` を返す。
fn position_final_score(score: Score, pos: &Position) -> Option<Score> {
    if pos.board().is_empty() {
        Some(score + SCORE_PERFECT)
    } else if !pos.has_action() {
        Some(score)
    } else {
        None
    }
}

/// 状態を評価関数にかけて `ChildState` を得る。
fn make_child_state(score: Score, pos: &Position, state: State) -> ChildState {
    // NOTE: とりあえず最終スコアの上界をそのまま使っておく。
    let eval = score + pos.gain_upper_bound();
    let eval = eval as u16;

    ChildState {
        parent_idx: state.parent_idx,
        parent_sq: state.parent_sq,
        eval,
        key: pos.key(),
    }
}

/// 1 本のビーム。
///
/// メモリ節約のため、`State` のフィールドを個別に `Vec` で持っている (Struct of Array)。
/// (`State` を pack するとアラインメントが崩れて遅そうなので...)
#[derive(Clone, Debug, Eq, PartialEq)]
struct Beam {
    parent_idxs: Vec<u32>,
    parent_sqs: Vec<Option<Square>>,
}

impl Beam {
    #[allow(dead_code)]
    fn new() -> Self {
        Self {
            parent_idxs: vec![],
            parent_sqs: vec![],
        }
    }

    fn with_capacity(capacity: usize) -> Self {
        Self {
            parent_idxs: Vec::with_capacity(capacity),
            parent_sqs: Vec::with_capacity(capacity),
        }
    }

    fn new_root() -> Self {
        let mut beam = Self::with_capacity(1);
        beam.push(State::new_root());
        beam
    }

    fn len(&self) -> usize {
        self.parent_idxs.len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn get(&self, i: usize) -> State {
        let parent_idx = self.parent_idxs[i];
        let parent_sq = self.parent_sqs[i];

        State {
            parent_idx,
            parent_sq,
        }
    }

    fn push(&mut self, state: State) {
        self.parent_idxs.push(state.parent_idx);
        self.parent_sqs.push(state.parent_sq);
    }

    fn enumerate(
        &self,
    ) -> impl ExactSizeIterator<Item = (usize, State)> + std::iter::FusedIterator + '_ {
        (0..self.len()).map(|i| (i, self.get(i)))
    }
}

/// ビームサーチにおける状態。
///
/// 親状態へのリンクだけを持つ。具体的な局面などはその都度経路復元して得る。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct State {
    /// 親状態のビーム内インデックス。
    parent_idx: u32,

    /// 親状態における着手マス。ルートノードの場合 `None`。
    parent_sq: Option<Square>,
}

const _: () = assert!(std::mem::size_of::<State>() == 8);

impl State {
    fn new_root() -> Self {
        Self {
            parent_idx: 0,
            parent_sq: None,
        }
    }

    fn new_child(parent_idx: usize, parent_sq: Square) -> Self {
        unsafe { assert_unchecked!(parent_idx <= u32::MAX as usize) }

        Self {
            parent_idx: parent_idx as u32,
            parent_sq: Some(parent_sq),
        }
    }

    fn parent(self) -> Option<(usize, Square)> {
        self.parent_sq
            .map(|parent_sq| (self.parent_idx as usize, parent_sq))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ChildState {
    parent_idx: u32,
    parent_sq: Option<Square>,
    eval: u16,
    key: u64,
}

const _: () = assert!(std::mem::size_of::<ChildState>() == 16);

impl ChildState {
    fn into_state(self) -> State {
        State {
            parent_idx: self.parent_idx,
            parent_sq: self.parent_sq,
        }
    }
}

impl PartialOrd for ChildState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ChildState {
    /// 評価値による比較。
    /// `ChildrenArray` 内のヒープを min-heap にするため、順序を逆にする(評価値が大きい方が先になる)。
    fn cmp(&self, other: &Self) -> Ordering {
        other.eval.cmp(&self.eval)
    }
}

/// 評価値上位 k 件の子状態を保持する。
/// 盤面が重複している場合、評価値が最大のものを優先する。
#[derive(Debug)]
struct Children {
    array: ChildrenArray,

    /// (盤面のハッシュ値, 最大評価値)。
    map: U64HashMap<u64, u16>,
}

impl Children {
    fn with_beam_capacity(capacity: usize) -> Self {
        Self {
            array: ChildrenArray::with_capacity(capacity),
            // 満杯になる頃にはそこまでエントリ数が増えなさそうなので、容量は少な目にしておく。
            map: u64_hashmap_with_capacity(4 * capacity),
        }
    }

    fn len(&self) -> usize {
        self.array.len()
    }

    #[allow(dead_code)]
    fn is_empty(&self) -> bool {
        self.array.is_empty()
    }

    /// 子状態 `x` の挿入を試みる。
    fn insert(&mut self, x: ChildState) {
        // 同一盤面かつ評価値が x.eval 以上の状態が既にあるなら弾く。
        let entry = self.map.entry(x.key);
        if let HashMapEntry::Occupied(occ) = &entry {
            if x.eval <= *occ.get() {
                return;
            }
        }

        // array への挿入を試みる。挿入されなかった場合は単に戻る。
        if !self.array.insert(x) {
            return;
        }

        // x の評価値を記録する。
        entry
            .and_modify(|eval| {
                chmax!(*eval, x.eval);
            })
            .or_insert(x.eval);
    }

    /// 次のビームに変換する。`self` は空になる(容量は変化しない)。
    ///
    /// 重複除去用の map のクリアも行う。
    fn drain_to_beam(&mut self) -> Beam {
        let mut beam = Beam::with_capacity(self.array.len());

        // 挿入順によっては重複を除去しきれていないことがある。
        // たとえば、同一盤面について評価値 (100, 150) の順に挿入した場合、
        // 評価値 100 の方も array 内に残留している可能性がある。
        //
        // よって、各子状態について改めて map 内の最大評価値を取得し、評価値がそれより小さいものは除く。
        let xs = self.array.drain().filter(|x| {
            let eval_max = *self.map.get(&x.key).unwrap();
            x.eval >= eval_max
        });
        for x in xs {
            beam.push(x.into_state());
        }

        self.map.clear();

        beam
    }
}

/// 評価値上位 k 件の子状態を保持する固定容量配列。
///
/// TODO: 満杯になるまでは単に配列に溜め、満杯になったら heapify する方が速いはず。
/// ただし enum で実装すると所有権周りがかなり面倒。二分ヒープを自前実装する方が多分楽。
#[derive(Debug)]
struct ChildrenArray {
    heap: BinaryHeap<ChildState>,
}

impl ChildrenArray {
    fn with_capacity(capacity: usize) -> Self {
        unsafe { assert_unchecked!(capacity != 0) }

        Self {
            heap: BinaryHeap::with_capacity(capacity),
        }
    }

    fn capacity(&self) -> usize {
        self.heap.capacity()
    }

    fn len(&self) -> usize {
        self.heap.len()
    }

    fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    fn is_full(&self) -> bool {
        self.len() == self.capacity()
    }

    /// 子状態 `x` の挿入を試み、実際に挿入されたかどうかを返す。
    ///
    /// 残り容量がなく、かつ `self` 内の最小の子状態の評価が `x` より悪い場合、
    /// その子状態を削除して `x` を挿入する。
    fn insert(&mut self, x: ChildState) -> bool {
        if self.is_full() {
            let x_peek = self.heap.peek().unwrap();
            if &x < x_peek {
                self.heap.pop().unwrap();
                self.heap.push(x);
                true
            } else {
                false
            }
        } else {
            self.heap.push(x);
            true
        }
    }

    /// `self` を空にし、中身を任意の順序で列挙する。
    /// `self` の容量は変化しない。
    fn drain(
        &mut self,
    ) -> impl DoubleEndedIterator<Item = ChildState> + ExactSizeIterator + std::iter::FusedIterator + '_
    {
        self.heap.drain()
    }
}

#[cfg(test)]
mod tests {
    use itertools::{assert_equal, Itertools as _};

    use super::*;

    fn make_child_eval_key(eval: u16, key: u64) -> ChildState {
        ChildState {
            parent_idx: 0,   // 値は適当
            parent_sq: None, // 値は適当
            eval,
            key,
        }
    }

    #[test]
    fn test_child_state_ord() {
        {
            let child = make_child_eval_key(1, 0x0);
            assert!(child <= child);
            assert!(child >= child);
        }

        assert!(make_child_eval_key(2, 0x0) < make_child_eval_key(1, 0x1));
        assert!(make_child_eval_key(1, 0x0) > make_child_eval_key(2, 0x1));
    }

    #[test]
    fn test_children_array() {
        const CAPACITY: usize = 3;

        let mut ary = ChildrenArray::with_capacity(CAPACITY);
        assert_eq!(ary.capacity(), CAPACITY);
        assert!(ary.is_empty());
        assert_eq!(ary.len(), 0);
        assert!(!ary.is_full());

        assert!(ary.insert(make_child_eval_key(2, 0x0)));
        assert!(!ary.is_empty());
        assert_eq!(ary.len(), 1);
        assert!(!ary.is_full());

        assert!(ary.insert(make_child_eval_key(3, 0x1)));
        assert!(!ary.is_empty());
        assert_eq!(ary.len(), 2);
        assert!(!ary.is_full());

        assert!(ary.insert(make_child_eval_key(4, 0x2)));
        assert!(!ary.is_empty());
        assert_eq!(ary.len(), CAPACITY);
        assert!(ary.is_full());

        assert!(!ary.insert(make_child_eval_key(1, 0x3)));
        assert!(!ary.is_empty());
        assert_eq!(ary.len(), CAPACITY);
        assert!(ary.is_full());

        assert!(ary.insert(make_child_eval_key(5, 0x4)));
        assert!(!ary.is_empty());
        assert_eq!(ary.len(), CAPACITY);
        assert!(ary.is_full());

        assert!(!ary.insert(make_child_eval_key(2, 0x5)));
        assert!(!ary.is_empty());
        assert_eq!(ary.len(), CAPACITY);
        assert!(ary.is_full());

        assert_equal(
            ary.drain().sorted(),
            [
                make_child_eval_key(5, 0x4),
                make_child_eval_key(4, 0x2),
                make_child_eval_key(3, 0x1),
            ],
        );

        assert_eq!(ary.capacity(), CAPACITY);
        assert!(ary.is_empty());
        assert_eq!(ary.len(), 0);
        assert!(!ary.is_full());
    }
}
