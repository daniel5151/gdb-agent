# gdb-agent

A Rust crate for handling gdb agent bytecode expressions.

Implements most bytecode features, with the notable exception of anything
tracing-related.

For more information on the agent expression mechanism, check out the docs at:
https://sourceware.org/gdb/onlinedocs/gdb/Agent-Expressions.html#Agent-Expressions

* * *

As of version `0.2.0`, this crate supports running in both `alloc`-only and
`no_std` projects, and is also guaranteed to be panic-free\*, making it suitable
for embedded use-cases.

\*assuming rustc successfully elides bounds checks, which requires compiling
with optimizations. If you notice this crate is introducing a panic, please file
a bug report!
