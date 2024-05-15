#![allow(unused_macros)]
#![allow(unused_imports)]

/// ${CARGO_MANIFEST_DIR}/asset/ 直下のファイルを `include!` する。
macro_rules! asset_include {
    ($file:expr) => {
        ::std::include!(::std::concat!(
            ::std::env!("CARGO_MANIFEST_DIR"),
            "/asset/",
            $file
        ))
    };
}
pub(crate) use asset_include;

/// ${CARGO_MANIFEST_DIR}/asset/ 直下のファイルを `include_bytes!` する。
#[allow(unused_macros)]
macro_rules! asset_include_bytes {
    ($file:expr) => {
        ::std::include_bytes!(::std::concat!(
            ::std::env!("CARGO_MANIFEST_DIR"),
            "/asset/",
            $file
        ))
    };
}
#[allow(unused_imports)]
pub(crate) use asset_include_bytes;
