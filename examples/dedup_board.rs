//! 有効な盤面を生成するパラメータの集合を重複なしで求める。

use std::collections::hash_map::Entry;

use samegame_sfc_large_2::*;

fn main() -> anyhow::Result<()> {
    let mut map = u64_hashmap_with_capacity::<u64, RandomBoardParam>(0x8000 * 0x100 * 5);

    for (param, board, legal, _rng_after) in enumerate_all_board() {
        if !legal {
            eprintln!("regen\t{param}");
            continue;
        }

        let pos = Position::new(board.clone());
        match map.entry(pos.key()) {
            Entry::Occupied(entry) => {
                let entry_param = entry.get();
                let (entry_board, _) = entry_param.gen_legal_board().unwrap();
                if entry_board == board {
                    eprintln!("duplicated\t{entry_param}\t{param}");
                } else {
                    eprintln!("collision\t{entry_param}\t{param}");
                    println!("{param}");
                }
            }
            Entry::Vacant(entry) => {
                println!("{param}");
                entry.insert(param);
            }
        }
    }

    Ok(())
}
