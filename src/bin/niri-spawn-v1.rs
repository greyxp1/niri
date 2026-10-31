#[path = "../spawn_helper.rs"]
mod spawn_helper;

fn main() -> anyhow::Result<()> {
    spawn_helper::spawn_child()
}
