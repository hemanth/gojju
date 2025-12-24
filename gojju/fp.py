"""
Functional Programming Library for Gojju.
Provides advanced FP constructs like Maybe, Either, Function composition, and more.
"""

from typing import Any, Callable, TypeVar, Generic, Optional
from functools import reduce as functools_reduce


# ============================================================================
# Maybe Monad
# ============================================================================

class Maybe:
    """
    Maybe monad for handling optional values.
    Inspired by Haskell's Maybe type.
    
    Usage:
        maybe = Maybe.just(5)
        maybe.map(lambda x: x * 2).get_or(0)  # 10
        
        nothing = Maybe.nothing()
        nothing.map(lambda x: x * 2).get_or(0)  # 0
    """
    
    def __init__(self, value: Any = None, is_nothing: bool = False):
        self._value = value
        self._is_nothing = is_nothing
    
    @classmethod
    def just(cls, value: Any) -> 'Maybe':
        """Create a Maybe with a value."""
        return cls(value, is_nothing=False)
    
    @classmethod
    def nothing(cls) -> 'Maybe':
        """Create an empty Maybe."""
        return cls(None, is_nothing=True)
    
    @classmethod
    def of(cls, value: Any) -> 'Maybe':
        """Create a Maybe, treating None as Nothing."""
        if value is None:
            return cls.nothing()
        return cls.just(value)
    
    def is_just(self) -> bool:
        return not self._is_nothing
    
    def is_nothing(self) -> bool:
        return self._is_nothing
    
    def map(self, fn: Callable) -> 'Maybe':
        """Apply function to value if present."""
        if self._is_nothing:
            return self
        return Maybe.just(fn(self._value))
    
    def flat_map(self, fn: Callable) -> 'Maybe':
        """Apply function that returns Maybe, flatten result."""
        if self._is_nothing:
            return self
        result = fn(self._value)
        if isinstance(result, Maybe):
            return result
        return Maybe.just(result)
    
    def filter(self, predicate: Callable) -> 'Maybe':
        """Keep value only if predicate is true."""
        if self._is_nothing:
            return self
        if predicate(self._value):
            return self
        return Maybe.nothing()
    
    def get(self) -> Any:
        """Get value or raise if Nothing."""
        if self._is_nothing:
            raise ValueError("Cannot get value from Nothing")
        return self._value
    
    def get_or(self, default: Any) -> Any:
        """Get value or return default."""
        if self._is_nothing:
            return default
        return self._value
    
    def get_or_else(self, fn: Callable) -> Any:
        """Get value or call function for default."""
        if self._is_nothing:
            return fn()
        return self._value
    
    def or_else(self, other: 'Maybe') -> 'Maybe':
        """Return self if Just, otherwise return other."""
        if self._is_nothing:
            return other
        return self
    
    def __repr__(self):
        if self._is_nothing:
            return "Nothing"
        return f"Just({self._value!r})"
    
    def __eq__(self, other):
        if not isinstance(other, Maybe):
            return False
        if self._is_nothing and other._is_nothing:
            return True
        if self._is_nothing or other._is_nothing:
            return False
        return self._value == other._value


# ============================================================================
# Either Monad
# ============================================================================

class Either:
    """
    Either monad for handling computations that may fail.
    Left represents failure, Right represents success.
    
    Usage:
        right = Either.right(5)
        right.map(lambda x: x * 2).get_or(0)  # 10
        
        left = Either.left("error")
        left.map(lambda x: x * 2).get_or(0)  # 0
    """
    
    def __init__(self, value: Any, is_left: bool = False):
        self._value = value
        self._is_left = is_left
    
    @classmethod
    def left(cls, value: Any) -> 'Either':
        """Create a Left (failure)."""
        return cls(value, is_left=True)
    
    @classmethod
    def right(cls, value: Any) -> 'Either':
        """Create a Right (success)."""
        return cls(value, is_left=False)
    
    @classmethod
    def try_of(cls, fn: Callable) -> 'Either':
        """Create Either from a function that might throw."""
        try:
            return cls.right(fn())
        except Exception as e:
            return cls.left(str(e))
    
    def is_left(self) -> bool:
        return self._is_left
    
    def is_right(self) -> bool:
        return not self._is_left
    
    def map(self, fn: Callable) -> 'Either':
        """Apply function to Right value."""
        if self._is_left:
            return self
        return Either.right(fn(self._value))
    
    def map_left(self, fn: Callable) -> 'Either':
        """Apply function to Left value."""
        if self._is_left:
            return Either.left(fn(self._value))
        return self
    
    def flat_map(self, fn: Callable) -> 'Either':
        """Apply function that returns Either, flatten result."""
        if self._is_left:
            return self
        result = fn(self._value)
        if isinstance(result, Either):
            return result
        return Either.right(result)
    
    def get(self) -> Any:
        """Get Right value or raise if Left."""
        if self._is_left:
            raise ValueError(f"Cannot get Right value from Left: {self._value}")
        return self._value
    
    def get_left(self) -> Any:
        """Get Left value or raise if Right."""
        if not self._is_left:
            raise ValueError("Cannot get Left value from Right")
        return self._value
    
    def get_or(self, default: Any) -> Any:
        """Get Right value or return default."""
        if self._is_left:
            return default
        return self._value
    
    def fold(self, left_fn: Callable, right_fn: Callable) -> Any:
        """Apply appropriate function based on Left/Right."""
        if self._is_left:
            return left_fn(self._value)
        return right_fn(self._value)
    
    def __repr__(self):
        if self._is_left:
            return f"Left({self._value!r})"
        return f"Right({self._value!r})"
    
    def __eq__(self, other):
        if not isinstance(other, Either):
            return False
        return self._is_left == other._is_left and self._value == other._value


# ============================================================================
# Lazy Evaluation
# ============================================================================

class Lazy:
    """
    Lazy evaluation wrapper.
    Delays computation until value is needed.
    
    Usage:
        lazy = Lazy(lambda: expensive_computation())
        lazy.get()  # Computes only when called
    """
    
    def __init__(self, thunk: Callable):
        self._thunk = thunk
        self._evaluated = False
        self._value = None
    
    def get(self) -> Any:
        """Force evaluation and return value."""
        if not self._evaluated:
            self._value = self._thunk()
            self._evaluated = True
        return self._value
    
    def map(self, fn: Callable) -> 'Lazy':
        """Create new Lazy with mapped value."""
        return Lazy(lambda: fn(self.get()))
    
    def flat_map(self, fn: Callable) -> 'Lazy':
        """Create new Lazy with flat-mapped value."""
        def thunk():
            result = fn(self.get())
            if isinstance(result, Lazy):
                return result.get()
            return result
        return Lazy(thunk)
    
    def __repr__(self):
        if self._evaluated:
            return f"Lazy({self._value!r})"
        return "Lazy(<unevaluated>)"


# ============================================================================
# Function Composition Utilities
# ============================================================================

def compose(*fns):
    """
    Compose functions right-to-left: compose(f, g, h)(x) = f(g(h(x)))
    """
    def composed(x):
        result = x
        for fn in reversed(fns):
            result = fn(result)
        return result
    return composed


def pipe(*fns):
    """
    Compose functions left-to-right: pipe(f, g, h)(x) = h(g(f(x)))
    """
    def piped(x):
        result = x
        for fn in fns:
            result = fn(result)
        return result
    return piped


def curry2(fn):
    """Curry a 2-argument function."""
    return lambda a: lambda b: fn(a, b)


def curry3(fn):
    """Curry a 3-argument function."""
    return lambda a: lambda b: lambda c: fn(a, b, c)


def uncurry2(fn):
    """Uncurry a curried 2-argument function."""
    return lambda a, b: fn(a)(b)


def uncurry3(fn):
    """Uncurry a curried 3-argument function."""
    return lambda a, b, c: fn(a)(b)(c)


def flip(fn):
    """Flip the order of arguments for a binary function."""
    return lambda a, b: fn(b, a)


def constant(value):
    """Create a function that always returns the same value."""
    return lambda *args: value


def identity(x):
    """Identity function."""
    return x


def partial(fn, *args):
    """Partially apply arguments to a function."""
    return lambda *more: fn(*args, *more)


def partial_right(fn, *args):
    """Partially apply arguments from the right."""
    return lambda *more: fn(*more, *args)


def memoize(fn):
    """Memoize a function (cache its results)."""
    cache = {}
    def memoized(*args):
        key = args
        if key not in cache:
            cache[key] = fn(*args)
        return cache[key]
    return memoized


def once(fn):
    """Create a function that only executes once."""
    called = False
    result = None
    def wrapper(*args):
        nonlocal called, result
        if not called:
            called = True
            result = fn(*args)
        return result
    return wrapper


def negate(predicate):
    """Negate a predicate function."""
    return lambda *args: not predicate(*args)


def complement(predicate):
    """Alias for negate."""
    return negate(predicate)


def juxt(*fns):
    """
    Create a function that applies multiple functions to the same argument.
    juxt(f, g, h)(x) = [f(x), g(x), h(x)]
    """
    return lambda x: [fn(x) for fn in fns]


def converge(fn, fns):
    """
    Apply multiple functions to an argument, then pass results to a combining function.
    converge(add, [double, triple])(5) = add(double(5), triple(5)) = add(10, 15) = 25
    """
    return lambda x: fn(*[f(x) for f in fns])


# ============================================================================
# List Utilities (Functional Style)
# ============================================================================

def foldl(fn, initial, lst):
    """Left fold (reduce from left)."""
    return functools_reduce(fn, lst, initial)


def foldr(fn, initial, lst):
    """Right fold (reduce from right)."""
    return functools_reduce(lambda acc, x: fn(x, acc), reversed(lst), initial)


def scanl(fn, initial, lst):
    """Left scan - like foldl but returns all intermediate values."""
    result = [initial]
    acc = initial
    for x in lst:
        acc = fn(acc, x)
        result.append(acc)
    return result


def scanr(fn, initial, lst):
    """Right scan - like foldr but returns all intermediate values."""
    result = [initial]
    acc = initial
    for x in reversed(lst):
        acc = fn(x, acc)
        result.insert(0, acc)
    return result


def unfold(fn, seed):
    """
    Build a list from a seed value using a generator function.
    fn returns None to stop, or (value, next_seed) to continue.
    """
    result = []
    current = seed
    while True:
        pair = fn(current)
        if pair is None:
            break
        value, current = pair
        result.append(value)
    return result


def iterate(fn, initial, n):
    """Generate n values by repeatedly applying fn."""
    result = [initial]
    current = initial
    for _ in range(n - 1):
        current = fn(current)
        result.append(current)
    return result


def repeat_value(value, n):
    """Repeat a value n times."""
    return [value] * n


def cycle(lst, n):
    """Cycle through a list n times."""
    result = []
    for i in range(n):
        result.append(lst[i % len(lst)])
    return result


def intersperse(separator, lst):
    """Place separator between each element."""
    if len(lst) <= 1:
        return lst[:]
    result = []
    for i, item in enumerate(lst):
        result.append(item)
        if i < len(lst) - 1:
            result.append(separator)
    return result


def intercalate(separator_list, lists):
    """Concatenate lists with separator list between them."""
    if not lists:
        return []
    result = lists[0][:]
    for lst in lists[1:]:
        result.extend(separator_list)
        result.extend(lst)
    return result


def partition(predicate, lst):
    """Split list into two: elements matching predicate and those that don't."""
    matching = []
    not_matching = []
    for item in lst:
        if predicate(item):
            matching.append(item)
        else:
            not_matching.append(item)
    return [matching, not_matching]


def group_by(fn, lst):
    """Group elements by the result of applying fn."""
    groups = {}
    for item in lst:
        key = fn(item)
        if key not in groups:
            groups[key] = []
        groups[key].append(item)
    return groups


def span(predicate, lst):
    """Split list at first element not matching predicate."""
    for i, item in enumerate(lst):
        if not predicate(item):
            return [lst[:i], lst[i:]]
    return [lst[:], []]


def break_on(predicate, lst):
    """Split list at first element matching predicate."""
    return span(lambda x: not predicate(x), lst)


def transpose(lists):
    """Transpose a list of lists."""
    if not lists:
        return []
    return [list(row) for row in zip(*lists)]


# ============================================================================
# Predicates
# ============================================================================

def is_even(n):
    return n % 2 == 0

def is_odd(n):
    return n % 2 == 1

def is_positive(n):
    return n > 0

def is_negative(n):
    return n < 0

def is_zero(n):
    return n == 0

def is_empty(collection):
    return len(collection) == 0

def is_not_empty(collection):
    return len(collection) > 0


# ============================================================================
# Register FP Utilities as Gojju Builtins
# ============================================================================

FP_BUILTINS = {
    # Monads
    'Maybe': Maybe,
    'Just': Maybe.just,
    'Nothing': Maybe.nothing,
    'Either': Either,
    'Left': Either.left,
    'Right': Either.right,
    'Lazy': Lazy,
    
    # Composition
    'compose': compose,
    'pipe': pipe,
    'curry2': curry2,
    'curry3': curry3,
    'uncurry2': uncurry2,
    'uncurry3': uncurry3,
    'flip': flip,
    'constant': constant,
    'identity': identity,
    'partial': partial,
    'partial_right': partial_right,
    'memoize': memoize,
    'once': once,
    'negate': negate,
    'complement': complement,
    'juxt': juxt,
    'converge': converge,
    
    # Folds
    'foldl': foldl,
    'foldr': foldr,
    'scanl': scanl,
    'scanr': scanr,
    
    # List generation
    'unfold': unfold,
    'iterate': iterate,
    'repeat_value': repeat_value,
    'cycle': cycle,
    'intersperse': intersperse,
    'intercalate': intercalate,
    
    # List partitioning
    'partition': partition,
    'group_by': group_by,
    'span': span,
    'break_on': break_on,
    'transpose': transpose,
    
    # Predicates
    'is_even': is_even,
    'is_odd': is_odd,
    'is_positive': is_positive,
    'is_negative': is_negative,
    'is_zero': is_zero,
    'is_empty': is_empty,
    'is_not_empty': is_not_empty,
}
