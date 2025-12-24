"""
Built-in functions for the Gojju programming language.
"""

import re
from typing import Any, List, Callable
import sys


class GojjuFunction:
    """Represents a built-in function."""
    def __init__(self, name: str, func: Callable, arity: int = -1):
        self.name = name
        self.func = func
        self.arity = arity  # -1 means variadic
    
    def __repr__(self):
        return f"<builtin {self.name}>"
    
    def __call__(self, *args):
        return self.func(*args)


class GojjuBlock:
    """Represents a Ruby-style block passed to a function."""
    def __init__(self, params: list, body, interpreter):
        self.params = params
        self.body = body
        self.interpreter = interpreter
    
    def __call__(self, *args):
        # Create a new scope for the block
        env = self.interpreter.env.copy()
        for i, param in enumerate(self.params):
            if i < len(args):
                env[param] = args[i]
            else:
                env[param] = None
        
        old_env = self.interpreter.env
        self.interpreter.env = env
        try:
            result = self.interpreter.evaluate(self.body)
            return result
        finally:
            self.interpreter.env = old_env


# ============================================================================
# IO Functions
# ============================================================================

def builtin_print(*args):
    """Print values to stdout."""
    print(*args)
    return None


def builtin_puts(*args):
    """Print values with newlines (Ruby-style)."""
    for arg in args:
        print(arg)
    return None


def builtin_gets():
    """Read a line from stdin."""
    return input()


def builtin_input(prompt=""):
    """Read input with an optional prompt."""
    return input(prompt)


# ============================================================================
# List Functions
# ============================================================================

def builtin_head(lst):
    """Get the first element of a list."""
    if not lst:
        return None
    return lst[0]


def builtin_tail(lst):
    """Get all but the first element of a list."""
    if not lst:
        return []
    return lst[1:]


def builtin_last(lst):
    """Get the last element of a list."""
    if not lst:
        return None
    return lst[-1]


def builtin_init(lst):
    """Get all but the last element."""
    if not lst:
        return []
    return lst[:-1]


def builtin_len(obj):
    """Get the length of a list or string."""
    return len(obj)


def builtin_push(lst, item):
    """Add an item to the end of a list (mutating)."""
    lst.append(item)
    return lst


def builtin_pop(lst):
    """Remove and return the last item."""
    if not lst:
        return None
    return lst.pop()


def builtin_concat(*lists):
    """Concatenate multiple lists."""
    result = []
    for lst in lists:
        if isinstance(lst, list):
            result.extend(lst)
        else:
            result.append(lst)
    return result


def builtin_reverse(lst):
    """Reverse a list."""
    return list(reversed(lst))


def builtin_sort(lst, key=None):
    """Sort a list."""
    return sorted(lst, key=key)


def builtin_unique(lst):
    """Remove duplicates from a list."""
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def builtin_flatten(lst):
    """Flatten a nested list by one level."""
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)
    return result


def builtin_zip(*lists):
    """Zip multiple lists together."""
    return [list(t) for t in zip(*lists)]


def builtin_enumerate(lst, start=0):
    """Return list of [index, value] pairs."""
    return [[i, v] for i, v in enumerate(lst, start)]


# ============================================================================
# Higher-Order Functions
# ============================================================================

def builtin_map(func, lst):
    """Apply a function to each element."""
    return [func(x) for x in lst]


def builtin_filter(func, lst):
    """Filter elements by predicate."""
    return [x for x in lst if func(x)]


def builtin_reduce(func, lst, initial=None):
    """Reduce a list to a single value."""
    if initial is not None:
        acc = initial
        items = lst
    else:
        if not lst:
            return None
        acc = lst[0]
        items = lst[1:]
    
    for item in items:
        acc = func(acc, item)
    return acc


def builtin_each(lst, func):
    """Iterate over each element (Ruby-style)."""
    for item in lst:
        func(item)
    return lst


def builtin_find(func, lst):
    """Find the first element matching predicate."""
    for item in lst:
        if func(item):
            return item
    return None


def builtin_any(func, lst):
    """Check if any element matches predicate."""
    for item in lst:
        if func(item):
            return True
    return False


def builtin_all(func, lst):
    """Check if all elements match predicate."""
    for item in lst:
        if not func(item):
            return False
    return True


def builtin_take(n, lst):
    """Take first n elements."""
    return lst[:n]


def builtin_drop(n, lst):
    """Drop first n elements."""
    return lst[n:]


def builtin_takewhile(func, lst):
    """Take elements while predicate is true."""
    result = []
    for item in lst:
        if func(item):
            result.append(item)
        else:
            break
    return result


def builtin_dropwhile(func, lst):
    """Drop elements while predicate is true."""
    result = []
    dropping = True
    for item in lst:
        if dropping and func(item):
            continue
        dropping = False
        result.append(item)
    return result


# ============================================================================
# String Functions
# ============================================================================

def builtin_split(s, sep=None):
    """Split a string."""
    return s.split(sep)


def builtin_join(lst, sep=""):
    """Join a list into a string."""
    return sep.join(str(x) for x in lst)


def builtin_upper(s):
    """Convert to uppercase."""
    return s.upper()


def builtin_lower(s):
    """Convert to lowercase."""
    return s.lower()


def builtin_capitalize(s):
    """Capitalize first letter."""
    return s.capitalize()


def builtin_strip(s):
    """Remove leading/trailing whitespace."""
    return s.strip()


def builtin_lstrip(s):
    """Remove leading whitespace."""
    return s.lstrip()


def builtin_rstrip(s):
    """Remove trailing whitespace."""
    return s.rstrip()


def builtin_replace(s, old, new):
    """Replace occurrences in a string."""
    return s.replace(old, new)


def builtin_startswith(s, prefix):
    """Check if string starts with prefix."""
    return s.startswith(prefix)


def builtin_endswith(s, suffix):
    """Check if string ends with suffix."""
    return s.endswith(suffix)


def builtin_contains(s, sub):
    """Check if string contains substring."""
    return sub in s


def builtin_chars(s):
    """Split string into characters."""
    return list(s)


def builtin_words(s):
    """Split string into words."""
    return s.split()


def builtin_lines(s):
    """Split string into lines."""
    return s.split('\n')


def builtin_match(pattern, s, flags=""):
    """Match regex pattern against string."""
    re_flags = 0
    if 'i' in flags:
        re_flags |= re.IGNORECASE
    if 'm' in flags:
        re_flags |= re.MULTILINE
    
    match = re.search(pattern, s, re_flags)
    if match:
        return match.group(0)
    return None


def builtin_matches(pattern, s, flags=""):
    """Find all regex matches."""
    re_flags = 0
    if 'i' in flags:
        re_flags |= re.IGNORECASE
    if 'm' in flags:
        re_flags |= re.MULTILINE
    
    return re.findall(pattern, s, re_flags)


def builtin_sub(pattern, replacement, s):
    """Replace first regex match."""
    return re.sub(pattern, replacement, s, count=1)


def builtin_gsub(pattern, replacement, s):
    """Replace all regex matches (global substitution)."""
    return re.sub(pattern, replacement, s)


# ============================================================================
# Math Functions
# ============================================================================

def builtin_abs(x):
    """Absolute value."""
    return abs(x)


def builtin_min(*args):
    """Minimum value."""
    if len(args) == 1 and isinstance(args[0], list):
        return min(args[0])
    return min(args)


def builtin_max(*args):
    """Maximum value."""
    if len(args) == 1 and isinstance(args[0], list):
        return max(args[0])
    return max(args)


def builtin_sum(lst):
    """Sum of a list."""
    return sum(lst)


def builtin_product(lst):
    """Product of a list."""
    result = 1
    for x in lst:
        result *= x
    return result


def builtin_range(*args):
    """Generate a range of numbers."""
    return list(range(*args))


def builtin_floor(x):
    """Floor of a number."""
    import math
    return math.floor(x)


def builtin_ceil(x):
    """Ceiling of a number."""
    import math
    return math.ceil(x)


def builtin_round(x, n=0):
    """Round a number."""
    return round(x, n)


def builtin_sqrt(x):
    """Square root."""
    import math
    return math.sqrt(x)


def builtin_pow(x, y):
    """Power."""
    return x ** y


# ============================================================================
# Type Functions
# ============================================================================

def builtin_type(x):
    """Get the type of a value."""
    if x is None:
        return "nil"
    if isinstance(x, bool):
        return "boolean"
    if isinstance(x, int):
        return "integer"
    if isinstance(x, float):
        return "float"
    if isinstance(x, str):
        return "string"
    if isinstance(x, list):
        return "list"
    if isinstance(x, dict):
        return "dict"
    if callable(x):
        return "function"
    return "unknown"


def builtin_int(x):
    """Convert to integer."""
    return int(x)


def builtin_float(x):
    """Convert to float."""
    return float(x)


def builtin_str(x):
    """Convert to string."""
    if x is None:
        return "nil"
    if isinstance(x, bool):
        return "true" if x else "false"
    return str(x)


def builtin_list(x):
    """Convert to list."""
    return list(x)


def builtin_bool(x):
    """Convert to boolean."""
    return bool(x)


# ============================================================================
# Utility Functions
# ============================================================================

def builtin_id(x):
    """Identity function."""
    return x


def builtin_compose(f, g):
    """Compose two functions: (compose f g)(x) = f(g(x))"""
    return lambda x: f(g(x))


def builtin_flip(f):
    """Flip argument order of a binary function."""
    return lambda a, b: f(b, a)


def builtin_const(x):
    """Create a function that always returns x."""
    return lambda _: x


def builtin_curry(f):
    """Curry a binary function."""
    return lambda x: lambda y: f(x, y)


def builtin_partial(f, *args):
    """Partially apply arguments to a function."""
    return lambda *more: f(*args, *more)


def builtin_times(n, func):
    """Execute a function n times."""
    results = []
    for i in range(n):
        results.append(func(i))
    return results


def builtin_assert(condition, message="Assertion failed"):
    """Assert a condition is true."""
    if not condition:
        raise AssertionError(message)
    return True


def builtin_exit(code=0):
    """Exit the program."""
    sys.exit(code)


# ============================================================================
# Built-in Registry
# ============================================================================

BUILTINS = {
    # IO
    'print': GojjuFunction('print', builtin_print),
    'puts': GojjuFunction('puts', builtin_puts),
    'gets': GojjuFunction('gets', builtin_gets, 0),
    'input': GojjuFunction('input', builtin_input),
    
    # List
    'head': GojjuFunction('head', builtin_head, 1),
    'tail': GojjuFunction('tail', builtin_tail, 1),
    'last': GojjuFunction('last', builtin_last, 1),
    'init': GojjuFunction('init', builtin_init, 1),
    'len': GojjuFunction('len', builtin_len, 1),
    'push': GojjuFunction('push', builtin_push, 2),
    'pop': GojjuFunction('pop', builtin_pop, 1),
    'concat': GojjuFunction('concat', builtin_concat),
    'reverse': GojjuFunction('reverse', builtin_reverse, 1),
    'sort': GojjuFunction('sort', builtin_sort),
    'unique': GojjuFunction('unique', builtin_unique, 1),
    'flatten': GojjuFunction('flatten', builtin_flatten, 1),
    'zip': GojjuFunction('zip', builtin_zip),
    'enumerate': GojjuFunction('enumerate', builtin_enumerate),
    
    # Higher-order
    'map': GojjuFunction('map', builtin_map, 2),
    'filter': GojjuFunction('filter', builtin_filter, 2),
    'reduce': GojjuFunction('reduce', builtin_reduce),
    'each': GojjuFunction('each', builtin_each, 2),
    'find': GojjuFunction('find', builtin_find, 2),
    'any': GojjuFunction('any', builtin_any, 2),
    'all': GojjuFunction('all', builtin_all, 2),
    'take': GojjuFunction('take', builtin_take, 2),
    'drop': GojjuFunction('drop', builtin_drop, 2),
    'takewhile': GojjuFunction('takewhile', builtin_takewhile, 2),
    'dropwhile': GojjuFunction('dropwhile', builtin_dropwhile, 2),
    
    # String
    'split': GojjuFunction('split', builtin_split),
    'join': GojjuFunction('join', builtin_join),
    'upper': GojjuFunction('upper', builtin_upper, 1),
    'lower': GojjuFunction('lower', builtin_lower, 1),
    'capitalize': GojjuFunction('capitalize', builtin_capitalize, 1),
    'strip': GojjuFunction('strip', builtin_strip, 1),
    'lstrip': GojjuFunction('lstrip', builtin_lstrip, 1),
    'rstrip': GojjuFunction('rstrip', builtin_rstrip, 1),
    'replace': GojjuFunction('replace', builtin_replace, 3),
    'startswith': GojjuFunction('startswith', builtin_startswith, 2),
    'endswith': GojjuFunction('endswith', builtin_endswith, 2),
    'contains': GojjuFunction('contains', builtin_contains, 2),
    'chars': GojjuFunction('chars', builtin_chars, 1),
    'words': GojjuFunction('words', builtin_words, 1),
    'lines': GojjuFunction('lines', builtin_lines, 1),
    'match': GojjuFunction('match', builtin_match),
    'matches': GojjuFunction('matches', builtin_matches),
    'sub': GojjuFunction('sub', builtin_sub, 3),
    'gsub': GojjuFunction('gsub', builtin_gsub, 3),
    
    # Math
    'abs': GojjuFunction('abs', builtin_abs, 1),
    'min': GojjuFunction('min', builtin_min),
    'max': GojjuFunction('max', builtin_max),
    'sum': GojjuFunction('sum', builtin_sum, 1),
    'product': GojjuFunction('product', builtin_product, 1),
    'range': GojjuFunction('range', builtin_range),
    'floor': GojjuFunction('floor', builtin_floor, 1),
    'ceil': GojjuFunction('ceil', builtin_ceil, 1),
    'round': GojjuFunction('round', builtin_round),
    'sqrt': GojjuFunction('sqrt', builtin_sqrt, 1),
    'pow': GojjuFunction('pow', builtin_pow, 2),
    
    # Type
    'type': GojjuFunction('type', builtin_type, 1),
    'int': GojjuFunction('int', builtin_int, 1),
    'float': GojjuFunction('float', builtin_float, 1),
    'str': GojjuFunction('str', builtin_str, 1),
    'list': GojjuFunction('list', builtin_list, 1),
    'bool': GojjuFunction('bool', builtin_bool, 1),
    
    # Utility
    'id': GojjuFunction('id', builtin_id, 1),
    'compose': GojjuFunction('compose', builtin_compose, 2),
    'flip': GojjuFunction('flip', builtin_flip, 1),
    'const': GojjuFunction('const', builtin_const, 1),
    'curry': GojjuFunction('curry', builtin_curry, 1),
    'partial': GojjuFunction('partial', builtin_partial),
    'times': GojjuFunction('times', builtin_times, 2),
    'assert': GojjuFunction('assert', builtin_assert),
    'exit': GojjuFunction('exit', builtin_exit),
}
