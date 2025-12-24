<p align="center">
  <img src="logo.png" alt="Gojju Logo" width="180">
</p>

<h1 align="center">Gojju 🌶️</h1>

<p align="center">
  <em>The essence of Python • Ruby • Haskell • Perl • JavaScript</em>
</p>

<p align="center">
  <a href="https://hemanth.github.io/gojju">Documentation</a> •
  <a href="https://hemanth.github.io/gojju/examples">Examples</a> •
  <a href="https://hemanth.github.io/gojju/api">API Reference</a>
</p>

---

**Gojju** (ಗೊಜ್ಜು) — meaning "essence" or "secret ingredient" in [Kannada](https://en.wikipedia.org/wiki/Kannada) — is a programming language combining the best of Python, Ruby, Haskell, Perl, and JavaScript.

## Installation

```bash
pip install gojju
```

## Quick Start

```bash
# Start REPL
gojju

# Run a file
gojju examples/hello.gj

# Execute inline
gojju -e "[1,2,3] |> map(\x -> x * 2) |> sum"
```

## Features at a Glance

| Source | What You Get |
|--------|--------------|
| 🐍 Python | List comprehensions, slicing |
| 💎 Ruby | `#{interpolation}`, postfix `if`, blocks |
| λ Haskell | `\|>` pipe, `\x -> x+1`, Maybe/Either |
| 🐪 Perl | `unless`/`until`, regex literals |
| ⚡ JS | Arrow functions `=>`, spread `...` |

## Example

```gojju
# Functional pipeline
[1, 2, 3, 4, 5]
  |> filter(\x -> x % 2 == 0)
  |> map(\x -> x * 2)
  |> sum

# Pattern matching
match value
  0 -> "zero"
  n if n < 0 -> "negative"
  _ -> "positive"
end

# Ruby-style string interpolation
let name = "World"
print "Hello, #{name}!"
```

## Documentation

📖 Full documentation, language guide, and API reference at **[hemanth.github.io/gojju](https://hemanth.github.io/gojju)**

## License

MIT
