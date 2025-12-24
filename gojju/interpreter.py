"""
Interpreter for the Gojju programming language.
Tree-walking interpreter that evaluates the AST.
"""

from typing import Any, Dict, Optional
import re
from .ast import *
from .builtins import BUILTINS, GojjuFunction, GojjuBlock
from .fp import FP_BUILTINS


class GojjuRuntimeError(Exception):
    """Runtime error during interpretation."""
    pass


class ReturnValue(Exception):
    """Used to implement return statements."""
    def __init__(self, value):
        self.value = value


class BreakLoop(Exception):
    """Used to implement break statements."""
    pass


class ContinueLoop(Exception):
    """Used to implement continue statements."""
    pass


class UserFunction:
    """A user-defined function."""
    def __init__(self, name: str, params: list, body: ASTNode, closure: Dict[str, Any], interpreter=None):
        self.name = name
        self.params = params
        self.body = body
        self.closure = closure
        self.interpreter = interpreter
    
    def __repr__(self):
        return f"<function {self.name}>"
    
    def __call__(self, *args):
        """Make UserFunction callable for use with builtins like map/filter."""
        if self.interpreter is None:
            raise GojjuRuntimeError("UserFunction has no interpreter reference")
        return self.interpreter._call_user_function(self, list(args))


class Interpreter:
    """Tree-walking interpreter for Gojju."""
    
    def __init__(self):
        self.env: Dict[str, Any] = {}
        self.mutables: set = set()  # Track mutable variables
        self._init_builtins()
    
    def _init_builtins(self):
        """Initialize built-in functions."""
        for name, func in BUILTINS.items():
            self.env[name] = func
        # Add FP utilities
        for name, func in FP_BUILTINS.items():
            self.env[name] = func
    
    def execute(self, node: ASTNode) -> Any:
        """Execute an AST node (top-level entry point)."""
        try:
            return self.evaluate(node)
        except ReturnValue as rv:
            return rv.value
    
    def evaluate(self, node: ASTNode) -> Any:
        """Evaluate an AST node."""
        method_name = f'eval_{type(node).__name__}'
        method = getattr(self, method_name, None)
        
        if method is None:
            raise GojjuRuntimeError(f"Unknown AST node type: {type(node).__name__}")
        
        return method(node)
    
    # ========================================================================
    # Literals
    # ========================================================================
    
    def eval_NumberLiteral(self, node: NumberLiteral) -> float:
        return node.value
    
    def eval_StringLiteral(self, node: StringLiteral) -> str:
        # Handle string interpolation
        result = node.value
        
        # Ruby-style interpolation: #{expr}
        def replace_ruby(match):
            expr_str = match.group(1)
            # This is a simplified version - full implementation would parse and eval
            if expr_str in self.env:
                return str(self.env[expr_str])
            return match.group(0)
        
        result = re.sub(r'#\{(\w+)\}', replace_ruby, result)
        
        # JS-style interpolation: ${expr}
        def replace_js(match):
            expr_str = match.group(1)
            if expr_str in self.env:
                return str(self.env[expr_str])
            return match.group(0)
        
        result = re.sub(r'\$\{(\w+)\}', replace_js, result)
        
        return result
    
    def eval_BooleanLiteral(self, node: BooleanLiteral) -> bool:
        return node.value
    
    def eval_NilLiteral(self, node: NilLiteral) -> None:
        return None
    
    def eval_SymbolLiteral(self, node: SymbolLiteral) -> str:
        # Symbols are represented as strings with a special prefix
        return f":{node.name}"
    
    def eval_RegexLiteral(self, node: RegexLiteral) -> re.Pattern:
        flags = 0
        if 'i' in node.flags:
            flags |= re.IGNORECASE
        if 'm' in node.flags:
            flags |= re.MULTILINE
        if 's' in node.flags:
            flags |= re.DOTALL
        if 'x' in node.flags:
            flags |= re.VERBOSE
        return re.compile(node.pattern, flags)
    
    def eval_ListLiteral(self, node: ListLiteral) -> list:
        result = []
        for elem in node.elements:
            if isinstance(elem, SpreadExpr):
                spread_val = self.evaluate(elem.expr)
                if isinstance(spread_val, list):
                    result.extend(spread_val)
                else:
                    raise GojjuRuntimeError("Spread operator requires a list")
            else:
                result.append(self.evaluate(elem))
        return result
    
    def eval_DictLiteral(self, node: DictLiteral) -> dict:
        result = {}
        for key, value in node.pairs:
            k = self.evaluate(key)
            v = self.evaluate(value)
            result[k] = v
        return result
    
    # ========================================================================
    # Expressions
    # ========================================================================
    
    def eval_Identifier(self, node: Identifier) -> Any:
        if node.name in self.env:
            return self.env[node.name]
        raise GojjuRuntimeError(f"Undefined variable: {node.name}")
    
    def eval_BinaryOp(self, node: BinaryOp) -> Any:
        # Short-circuit evaluation for and/or
        if node.operator == 'and':
            left = self.evaluate(node.left)
            if not self._is_truthy(left):
                return left
            return self.evaluate(node.right)
        
        if node.operator == 'or':
            left = self.evaluate(node.left)
            if self._is_truthy(left):
                return left
            return self.evaluate(node.right)
        
        left = self.evaluate(node.left)
        right = self.evaluate(node.right)
        
        ops = {
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            '/': lambda a, b: a / b,
            '%': lambda a, b: a % b,
            '**': lambda a, b: a ** b,
            '==': lambda a, b: a == b,
            '!=': lambda a, b: a != b,
            '<': lambda a, b: a < b,
            '>': lambda a, b: a > b,
            '<=': lambda a, b: a <= b,
            '>=': lambda a, b: a >= b,
            '..': lambda a, b: list(range(int(a), int(b) + 1)),
        }
        
        if node.operator in ops:
            try:
                return ops[node.operator](left, right)
            except Exception as e:
                raise GojjuRuntimeError(f"Error in {node.operator}: {e}")
        
        raise GojjuRuntimeError(f"Unknown operator: {node.operator}")
    
    def eval_UnaryOp(self, node: UnaryOp) -> Any:
        operand = self.evaluate(node.operand)
        
        if node.operator == '-':
            return -operand
        if node.operator == 'not':
            return not self._is_truthy(operand)
        
        raise GojjuRuntimeError(f"Unknown unary operator: {node.operator}")
    
    def eval_Call(self, node: Call) -> Any:
        callee = self.evaluate(node.callee)
        args = [self.evaluate(arg) for arg in node.arguments]
        
        return self._call_function(callee, args)
    
    def eval_MethodCall(self, node: MethodCall) -> Any:
        obj = self.evaluate(node.object)
        args = [self.evaluate(arg) for arg in node.arguments]
        
        # Look for method on object type
        if isinstance(obj, list):
            return self._list_method(obj, node.method, args)
        if isinstance(obj, str):
            return self._string_method(obj, node.method, args)
        if isinstance(obj, dict):
            return self._dict_method(obj, node.method, args)
        
        raise GojjuRuntimeError(f"No method '{node.method}' on {type(obj).__name__}")
    
    def _list_method(self, lst: list, method: str, args: list) -> Any:
        """Handle list methods."""
        methods = {
            'push': lambda: (lst.append(args[0]), lst)[1],
            'pop': lambda: lst.pop() if lst else None,
            'len': lambda: len(lst),
            'first': lambda: lst[0] if lst else None,
            'last': lambda: lst[-1] if lst else None,
            'empty?': lambda: len(lst) == 0,
            'reverse': lambda: list(reversed(lst)),
            'sort': lambda: sorted(lst),
            'join': lambda: (args[0] if args else "").join(str(x) for x in lst),
            'include?': lambda: args[0] in lst,
            'index': lambda: lst.index(args[0]) if args[0] in lst else None,
            'clear': lambda: (lst.clear(), lst)[1],
            'each': lambda: self._list_each(lst, args[0]) if args else lst,
            'map': lambda: [self._call_function(args[0], [x]) for x in lst] if args else lst,
            'filter': lambda: [x for x in lst if self._call_function(args[0], [x])] if args else lst,
            'select': lambda: [x for x in lst if self._call_function(args[0], [x])] if args else lst,
            'reject': lambda: [x for x in lst if not self._call_function(args[0], [x])] if args else lst,
            'reduce': lambda: self._list_reduce(lst, args) if args else None,
            'find': lambda: next((x for x in lst if self._call_function(args[0], [x])), None) if args else None,
            'any?': lambda: any(self._call_function(args[0], [x]) for x in lst) if args else any(lst),
            'all?': lambda: all(self._call_function(args[0], [x]) for x in lst) if args else all(lst),
            'take': lambda: lst[:args[0]] if args else lst,
            'drop': lambda: lst[args[0]:] if args else lst,
            'compact': lambda: [x for x in lst if x is not None],
            'flatten': lambda: self._flatten_list(lst),
            'uniq': lambda: list(dict.fromkeys(lst)),
            'sum': lambda: sum(lst),
            'min': lambda: min(lst) if lst else None,
            'max': lambda: max(lst) if lst else None,
        }
        
        if method in methods:
            return methods[method]()
        
        raise GojjuRuntimeError(f"Unknown list method: {method}")
    
    def _string_method(self, s: str, method: str, args: list) -> Any:
        """Handle string methods."""
        methods = {
            'len': lambda: len(s),
            'upper': lambda: s.upper(),
            'lower': lambda: s.lower(),
            'capitalize': lambda: s.capitalize(),
            'strip': lambda: s.strip(),
            'lstrip': lambda: s.lstrip(),
            'rstrip': lambda: s.rstrip(),
            'split': lambda: s.split(args[0] if args else None),
            'replace': lambda: s.replace(args[0], args[1]) if len(args) >= 2 else s,
            'startswith?': lambda: s.startswith(args[0]) if args else False,
            'endswith?': lambda: s.endswith(args[0]) if args else False,
            'include?': lambda: args[0] in s if args else False,
            'contains?': lambda: args[0] in s if args else False,
            'empty?': lambda: len(s) == 0,
            'chars': lambda: list(s),
            'lines': lambda: s.split('\n'),
            'words': lambda: s.split(),
            'reverse': lambda: s[::-1],
            'repeat': lambda: s * (args[0] if args else 1),
            'index': lambda: s.find(args[0]) if args and args[0] in s else None,
            'match': lambda: bool(re.search(args[0], s)) if args else False,
            'sub': lambda: re.sub(args[0], args[1], s, count=1) if len(args) >= 2 else s,
            'gsub': lambda: re.sub(args[0], args[1], s) if len(args) >= 2 else s,
        }
        
        if method in methods:
            return methods[method]()
        
        raise GojjuRuntimeError(f"Unknown string method: {method}")
    
    def _dict_method(self, d: dict, method: str, args: list) -> Any:
        """Handle dictionary methods."""
        methods = {
            'keys': lambda: list(d.keys()),
            'values': lambda: list(d.values()),
            'items': lambda: [[k, v] for k, v in d.items()],
            'get': lambda: d.get(args[0], args[1] if len(args) > 1 else None) if args else None,
            'has?': lambda: args[0] in d if args else False,
            'empty?': lambda: len(d) == 0,
            'len': lambda: len(d),
            'merge': lambda: {**d, **args[0]} if args else d,
            'delete': lambda: (d.pop(args[0], None), d)[1] if args else d,
            'clear': lambda: (d.clear(), d)[1],
        }
        
        if method in methods:
            return methods[method]()
        
        raise GojjuRuntimeError(f"Unknown dict method: {method}")
    
    def _list_each(self, lst: list, func) -> list:
        """Iterate over list with function."""
        for item in lst:
            self._call_function(func, [item])
        return lst
    
    def _list_reduce(self, lst: list, args: list) -> Any:
        """Reduce a list."""
        if not args:
            return None
        
        func = args[0]
        if len(args) > 1:
            acc = args[1]
            items = lst
        else:
            if not lst:
                return None
            acc = lst[0]
            items = lst[1:]
        
        for item in items:
            acc = self._call_function(func, [acc, item])
        return acc
    
    def _flatten_list(self, lst: list) -> list:
        """Flatten a nested list."""
        result = []
        for item in lst:
            if isinstance(item, list):
                result.extend(item)
            else:
                result.append(item)
        return result
    
    def _call_function(self, callee: Any, args: list) -> Any:
        """Call a function with arguments."""
        if isinstance(callee, GojjuFunction):
            return callee(*args)
        
        if isinstance(callee, UserFunction):
            return self._call_user_function(callee, args)
        
        if isinstance(callee, GojjuBlock):
            return callee(*args)
        
        if callable(callee):
            return callee(*args)
        
        raise GojjuRuntimeError(f"Cannot call non-function: {type(callee).__name__}")
    
    def _call_user_function(self, func: UserFunction, args: list) -> Any:
        """Call a user-defined function."""
        # Create new environment with closure
        env = func.closure.copy()
        
        # Bind parameters
        for i, param in enumerate(func.params):
            if i < len(args):
                env[param] = args[i]
            else:
                env[param] = None
        
        # Execute body
        old_env = self.env
        old_mutables = self.mutables
        self.env = env
        self.mutables = set()
        
        try:
            result = self.evaluate(func.body)
            return result
        except ReturnValue as rv:
            return rv.value
        finally:
            self.env = old_env
            self.mutables = old_mutables
    
    def eval_Index(self, node: Index) -> Any:
        obj = self.evaluate(node.object)
        idx = self.evaluate(node.index)
        
        try:
            return obj[idx]
        except (IndexError, KeyError) as e:
            return None
    
    def eval_Slice(self, node: Slice) -> Any:
        obj = self.evaluate(node.object)
        start = self.evaluate(node.start) if node.start else None
        end = self.evaluate(node.end) if node.end else None
        step = self.evaluate(node.step) if node.step else None
        
        if start is not None:
            start = int(start)
        if end is not None:
            end = int(end)
        if step is not None:
            step = int(step)
        
        return obj[start:end:step]
    
    def eval_PropertyAccess(self, node: PropertyAccess) -> Any:
        obj = self.evaluate(node.object)
        
        if node.optional and obj is None:
            return None
        
        if isinstance(obj, dict):
            return obj.get(node.property)
        
        raise GojjuRuntimeError(f"Cannot access property '{node.property}' on {type(obj).__name__}")
    
    def eval_Lambda(self, node: Lambda) -> UserFunction:
        return UserFunction('<lambda>', node.params, node.body, self.env.copy(), self)
    
    def eval_PipeExpr(self, node: PipeExpr) -> Any:
        value = self.evaluate(node.value)
        
        # If the function is a Call expression, inject value as last argument
        if isinstance(node.function, Call):
            callee = self.evaluate(node.function.callee)
            args = [self.evaluate(arg) for arg in node.function.arguments] + [value]
            return self._call_function(callee, args)
        
        # Otherwise, just call the function with the value
        func = self.evaluate(node.function)
        return self._call_function(func, [value])
    
    def eval_TernaryExpr(self, node: TernaryExpr) -> Any:
        condition = self.evaluate(node.condition)
        if self._is_truthy(condition):
            return self.evaluate(node.then_expr)
        return self.evaluate(node.else_expr)
    
    def eval_ListComprehension(self, node: ListComprehension) -> list:
        iterable = self.evaluate(node.iterable)
        result = []
        
        for item in iterable:
            self.env[node.var] = item
            
            if node.condition:
                if not self._is_truthy(self.evaluate(node.condition)):
                    continue
            
            result.append(self.evaluate(node.expr))
        
        return result
    
    def eval_SpreadExpr(self, node: SpreadExpr) -> Any:
        return self.evaluate(node.expr)
    
    # ========================================================================
    # Statements
    # ========================================================================
    
    def eval_LetStatement(self, node: LetStatement) -> Any:
        value = self.evaluate(node.value)
        self.env[node.name] = value
        
        if node.mutable:
            self.mutables.add(node.name)
        
        return value
    
    def eval_Assignment(self, node: Assignment) -> Any:
        value = self.evaluate(node.value)
        
        if isinstance(node.target, Identifier):
            name = node.target.name
            if name not in self.env:
                raise GojjuRuntimeError(f"Undefined variable: {name}")
            if name not in self.mutables and name not in BUILTINS:
                # Check if it's a global that was declared with let (immutable)
                # For simplicity, we'll allow reassignment in REPL mode
                pass
            self.env[name] = value
        elif isinstance(node.target, Index):
            obj = self.evaluate(node.target.object)
            idx = self.evaluate(node.target.index)
            obj[idx] = value
        elif isinstance(node.target, PropertyAccess):
            obj = self.evaluate(node.target.object)
            if isinstance(obj, dict):
                obj[node.target.property] = value
            else:
                raise GojjuRuntimeError(f"Cannot set property on {type(obj).__name__}")
        else:
            raise GojjuRuntimeError("Invalid assignment target")
        
        return value
    
    def eval_IfExpr(self, node: IfExpr) -> Any:
        condition = self.evaluate(node.condition)
        
        if node.is_unless:
            condition = not self._is_truthy(condition)
        else:
            condition = self._is_truthy(condition)
        
        if condition:
            return self.evaluate(node.then_branch)
        elif node.else_branch:
            return self.evaluate(node.else_branch)
        
        return None
    
    def eval_WhileLoop(self, node: WhileLoop) -> Any:
        result = None
        
        while True:
            condition = self.evaluate(node.condition)
            
            if node.is_until:
                condition = not self._is_truthy(condition)
            else:
                condition = self._is_truthy(condition)
            
            if not condition:
                break
            
            try:
                result = self.evaluate(node.body)
            except BreakLoop:
                break
            except ContinueLoop:
                continue
        
        return result
    
    def eval_ForLoop(self, node: ForLoop) -> Any:
        iterable = self.evaluate(node.iterable)
        result = None
        
        for item in iterable:
            self.env[node.var] = item
            
            try:
                result = self.evaluate(node.body)
            except BreakLoop:
                break
            except ContinueLoop:
                continue
        
        return result
    
    def eval_MatchExpr(self, node: MatchExpr) -> Any:
        value = self.evaluate(node.value)
        
        for case in node.cases:
            if self._match_pattern(case.pattern, value):
                if case.guard:
                    if not self._is_truthy(self.evaluate(case.guard)):
                        continue
                return self.evaluate(case.body)
        
        return None
    
    def _match_pattern(self, pattern: ASTNode, value: Any) -> bool:
        """Check if a value matches a pattern."""
        if isinstance(pattern, Wildcard):
            return True
        
        if isinstance(pattern, Identifier):
            # Bind the value to the identifier
            self.env[pattern.name] = value
            return True
        
        if isinstance(pattern, NumberLiteral):
            return value == pattern.value
        
        if isinstance(pattern, StringLiteral):
            return value == pattern.value
        
        if isinstance(pattern, BooleanLiteral):
            return value == pattern.value
        
        if isinstance(pattern, NilLiteral):
            return value is None
        
        if isinstance(pattern, ListLiteral):
            if not isinstance(value, list):
                return False
            if len(pattern.elements) != len(value):
                return False
            return all(self._match_pattern(p, v) for p, v in zip(pattern.elements, value))
        
        # Fall back to equality check
        try:
            return self.evaluate(pattern) == value
        except:
            return False
    
    def eval_FunctionDef(self, node: FunctionDef) -> UserFunction:
        func = UserFunction(node.name, node.params, node.body, self.env.copy(), self)
        self.env[node.name] = func
        return func
    
    def eval_ReturnStatement(self, node: ReturnStatement) -> Any:
        value = None
        if node.value:
            value = self.evaluate(node.value)
        raise ReturnValue(value)
    
    def eval_BreakStatement(self, node: BreakStatement) -> None:
        raise BreakLoop()
    
    def eval_ContinueStatement(self, node: ContinueStatement) -> None:
        raise ContinueLoop()
    
    def eval_Block(self, node: Block) -> Any:
        result = None
        for stmt in node.statements:
            result = self.evaluate(stmt)
        return result
    
    def eval_Program(self, node: Program) -> Any:
        result = None
        for stmt in node.statements:
            result = self.evaluate(stmt)
        return result
    
    def eval_BlockWithParams(self, node: BlockWithParams) -> GojjuBlock:
        return GojjuBlock(node.params, node.body, self)
    
    def eval_PostfixIf(self, node: PostfixIf) -> Any:
        condition = self.evaluate(node.condition)
        
        if node.is_unless:
            condition = not self._is_truthy(condition)
        else:
            condition = self._is_truthy(condition)
        
        if condition:
            return self.evaluate(node.expr)
        
        return None
    
    def eval_MatchCase(self, node: MatchCase) -> Any:
        # This should not be called directly
        raise GojjuRuntimeError("MatchCase should not be evaluated directly")
    
    def eval_Wildcard(self, node: Wildcard) -> None:
        return None
    
    # ========================================================================
    # Helpers
    # ========================================================================
    
    def _is_truthy(self, value: Any) -> bool:
        """Determine if a value is truthy (Perl/Ruby-style)."""
        if value is None:
            return False
        if value is False:
            return False
        if value == 0:
            return False
        if value == "":
            return False
        if isinstance(value, list) and len(value) == 0:
            return False
        return True
