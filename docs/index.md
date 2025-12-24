---
layout: default
title: Gojju - The Essence of Five Languages
---

# Gojju 🌶️

**Gojju** (ಗೊಜ್ಜು) — meaning "essence" or "secret ingredient" in [Kannada](https://en.wikipedia.org/wiki/Kannada) — is a programming language that combines the best features from Python, Ruby, Haskell, Perl, and JavaScript into one expressive, fun, and functional language.

## Installation

```bash
pip install gojju
```

## Quick Start

### REPL

```bash
gojju
```

```
╔══════════════════════════════════════════════════════════════╗
║   ██████╗  ██████╗      ██╗     ██╗██╗   ██╗                 ║
║  ██╔════╝ ██╔═══██╗     ██║     ██║██║   ██║                 ║
║  ██║  ███╗██║   ██║     ██║     ██║██║   ██║                 ║
║  ██║   ██║██║   ██║██   ██║██   ██║██║   ██║                 ║
║  ╚██████╔╝╚██████╔╝╚█████╔╝╚█████╔╝╚██████╔╝                 ║
╚══════════════════════════════════════════════════════════════╝

gojju> let x = 42
gojju> x * 2
=> 84
```

### Run a File

```bash
gojju script.gj
```

### Execute Inline

```bash
gojju -e "[1,2,3] |> map(\x -> x * 2) |> sum"
# Output: 12
```

## Language Features

### From Python 🐍

- **List comprehensions**: `[x * x for x in range(10) if x % 2 == 0]`
- **Slicing**: `list[1:3]`, `list[::2]`
- **Clean, readable syntax**

### From Ruby 💎

- **String interpolation**: `"Hello #{name}!"`
- **Implicit returns**: Last expression is the return value
- **Postfix conditionals**: `print "yes" if happy`
- **Blocks**: `list.each do |x| print x end`
- **Symbols**: `:success`, `:error`

### From Haskell λ

- **Immutable by default**: Use `let` for immutable, `mut` for mutable
- **Pattern matching**: With guards and wildcards
- **Pipe operator**: `value |> function`
- **Lambda syntax**: `\x -> x * 2`
- **Maybe/Either monads**: For safe computations
- **Function composition**: `compose`, `pipe`

### From Perl 🐪

- **`unless`/`until`**: Negated conditionals
- **Regex literals**: `/pattern/flags`
- **Powerful string manipulation**

### From JavaScript ⚡

- **Arrow functions**: `(x) => x * 2`
- **Spread operator**: `[...list1, ...list2]`
- **Optional chaining**: `obj?.property`

## Syntax Guide

### Variables

```gojju
# Immutable (default)
let name = "Gojju"
let pi = 3.14159

# Mutable
mut counter = 0
counter = counter + 1
```

### Functions

```gojju
# Function definition
fn greet(name)
  "Hello, #{name}!"
end

# Arrow function
let double = (x) => x * 2

# Haskell-style lambda
let triple = \x -> x * 3

# Multi-parameter lambda
let add = \a b -> a + b
```

### Control Flow

```gojju
# If expression
let result = if x > 0
  "positive"
else if x < 0
  "negative"
else
  "zero"
end

# Unless (negated if)
unless error
  continue_processing()
end

# Postfix conditional
print "success" if passed
print "warning" unless safe

# While loop
while count < 10
  count = count + 1
end

# Until loop (negated while)
until done
  process_next()
end

# For loop
for item in collection
  print item
end
```

### Pattern Matching

```gojju
match value
  0 -> "zero"
  1 -> "one"
  n if n < 0 -> "negative"
  n if n < 10 -> "small"
  _ -> "large"
end
```

### Lists & Comprehensions

```gojju
let numbers = [1, 2, 3, 4, 5]

# List comprehension
let squares = [x * x for x in range(10)]
let evens = [x for x in numbers if x % 2 == 0]

# List operations
head(numbers)      # 1
tail(numbers)      # [2, 3, 4, 5]
numbers[0]         # 1
numbers[1:3]       # [2, 3]
```

### Functional Programming

```gojju
# Pipe operator
[1, 2, 3, 4, 5]
  |> filter(\x -> x % 2 == 0)
  |> map(\x -> x * 2)
  |> sum

# Higher-order functions
map(\x -> x * 2, [1, 2, 3])           # [2, 4, 6]
filter(\x -> x > 2, [1, 2, 3, 4])     # [3, 4]
reduce(\acc x -> acc + x, [1,2,3], 0) # 6

# Function composition
let f = compose(double, increment)  # f(x) = double(increment(x))

# Currying
let add = \a b -> a + b
let add5 = curry2(add)(5)
add5(10)  # 15

# Monads
Just(5).map(\x -> x * 2).get_or(0)   # 10
Nothing().map(\x -> x * 2).get_or(0) # 0
```

## Built-in Functions

### I/O
- `print(*args)` - Print values
- `puts(*args)` - Print with newlines
- `input(prompt)` - Read input

### Lists
- `head(list)`, `tail(list)`, `last(list)`, `init(list)`
- `len(list)`, `reverse(list)`, `sort(list)`
- `push(list, item)`, `pop(list)`
- `map(fn, list)`, `filter(fn, list)`, `reduce(fn, list, init)`
- `take(n, list)`, `drop(n, list)`
- `find(fn, list)`, `any(fn, list)`, `all(fn, list)`

### Strings
- `upper(s)`, `lower(s)`, `capitalize(s)`
- `strip(s)`, `split(s, sep)`, `join(list, sep)`
- `replace(s, old, new)`
- `match(pattern, s)`, `sub(pattern, repl, s)`, `gsub(pattern, repl, s)`

### Math
- `abs(x)`, `min(...)`, `max(...)`, `sum(list)`
- `sqrt(x)`, `pow(x, y)`, `floor(x)`, `ceil(x)`, `round(x)`
- `range(start, end, step)`

### Functional
- `compose(*fns)`, `pipe(*fns)`
- `curry2(fn)`, `curry3(fn)`
- `partial(fn, *args)`, `flip(fn)`
- `identity(x)`, `constant(x)`
- `foldl(fn, init, list)`, `foldr(fn, init, list)`
- `scanl(fn, init, list)`, `scanr(fn, init, list)`

### Monads
- `Just(value)`, `Nothing()`
- `Left(value)`, `Right(value)`
- `Lazy(thunk)`

### Type
- `type(x)` - Get type name
- `int(x)`, `float(x)`, `str(x)`, `list(x)`, `bool(x)` - Conversions

## Examples

See the [examples directory](https://github.com/hemanth/gojju/tree/main/examples) for more:

- [Hello World](https://github.com/hemanth/gojju/blob/main/examples/hello.gj)
- [Functional Programming](https://github.com/hemanth/gojju/blob/main/examples/fp.gj)
- [Pattern Matching](https://github.com/hemanth/gojju/blob/main/examples/pattern_matching.gj)
