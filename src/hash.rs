//! zobrist hash を標準の `HashSet`, `HashMap` で使うためのアダプタ。
//!
//! set, map のキー型は `Hash::hash()` 内で自身の zobrist hash 値を `Hasher::write_u64()` に渡すこと。
//! (`u64` 型はこの条件を満たす)

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

/// zobrist hash 値をそのままキーとして使える set。
pub type U64HashSet<T> = HashSet<T, BuildHasherDefault<U64Hasher>>;

pub fn u64_hashset_with_capacity<T>(capacity: usize) -> U64HashSet<T> {
    U64HashSet::with_capacity_and_hasher(capacity, Default::default())
}

/// zobrist hash 値をそのままキーとして使える map。
pub type U64HashMap<K, V> = HashMap<K, V, BuildHasherDefault<U64Hasher>>;

pub fn u64_hashmap_with_capacity<K, V>(capacity: usize) -> U64HashMap<K, V> {
    U64HashMap::with_capacity_and_hasher(capacity, Default::default())
}

#[repr(transparent)]
#[derive(Debug, Default)]
pub struct U64Hasher(u64);

impl Hasher for U64Hasher {
    fn write(&mut self, _: &[u8]) {
        unimplemented!();
    }

    fn write_u64(&mut self, x: u64) {
        self.0 = x;
    }

    fn finish(&self) -> u64 {
        self.0
    }
}
