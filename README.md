# Tof

Completely experimental stuff, not even a prototype yet. Just a place to try out some ideas.

`rm -rf vault_test_manual`
`cargo run --release --manifest-path coldstorage/Cargo.toml -- ingest dataset/1AAEE0DE7F7A5D358956A2F825ED9A81.JPG --storage ./vault_test_manual --quality 8`
`cargo run --release --manifest-path coldstorage/Cargo.toml -- ingest dataset --storage ./vault_test_manual --quality 8`
`cargo run --release --manifest-path coldstorage/Cargo.toml -- retrieve 1 --storage ./vault_test_manual --quality 8`
`cargo run --release --manifest-path coldstorage/Cargo.toml -- stats --storage ./vault_test_manual --quality 8`
`cargo run --release --manifest-path coldstorage/Cargo.toml -- finetune --storage ./vault_test_manual --quality 8`