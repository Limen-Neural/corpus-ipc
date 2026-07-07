# Code Review Guide

## PR Checklist

- [ ] `cargo fmt --check` passes
- [ ] `cargo clippy --all-targets --all-features -- -D warnings` passes
- [ ] `cargo test --all-features` passes
- [ ] No new warnings

## Wire Protocol Changes

- Adding `#[serde(alias = "...")]` preserves backward compatibility for deserialization
- Renaming enum variants in `RuntimeMessage` is safe (externally-tagged — serde uses variant names)
- Renaming struct fields changes the serialization output — requires downstream coordination
- Do NOT rename literature citations in doc comments without verifying the actual paper title

## Naming Conventions

- Use generic runtime/IPC terminology (not domain-specific)
- Current names: `RuntimeBackend`, `RuntimeSnapshot`, `RuntimeMessage`, `process_batch`
- Retired names: `NeroManifoldSnapshot`, `SpineMessage`, `ZmqBrainBackend`, `process_signals`, `brain_tick`

## Bot Review Replies

- Reply to ALL review threads — never skip bots
- Reply substantively — never empty acknowledgments
- Use GraphQL `addPullRequestReviewThreadReply` (REST `POST .../replies` silently fails)
- After fixing, resolve threads via `resolveReviewThread` mutation
