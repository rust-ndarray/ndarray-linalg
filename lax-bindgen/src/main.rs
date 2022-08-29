use anyhow::Result;
use std::{fs, path::PathBuf};

fn main() -> Result<()> {
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let lapack_sys = fs::read_to_string(crate_root.join("lapack-sys/src/lapack.rs"))?;

    let file: syn::File = syn::parse_str(&lapack_sys)?;
    dbg!(file);

    Ok(())
}
