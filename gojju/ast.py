"""
Abstract Syntax Tree (AST) node definitions for Gojju.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


# Base class for all AST nodes
@dataclass
class ASTNode:
    """Base class for all AST nodes."""
    pass


# ============================================================================
# Literals
# ============================================================================

@dataclass
class NumberLiteral(ASTNode):
    """Numeric literal (int or float)."""
    value: float
    
    def __repr__(self):
        return f"Number({self.value})"


@dataclass
class StringLiteral(ASTNode):
    """String literal with potential interpolation."""
    value: str
    interpolations: list = field(default_factory=list)
    
    def __repr__(self):
        return f"String({self.value!r})"


@dataclass
class BooleanLiteral(ASTNode):
    """Boolean literal (true/false)."""
    value: bool
    
    def __repr__(self):
        return f"Bool({self.value})"


@dataclass
class NilLiteral(ASTNode):
    """Nil/null literal."""
    
    def __repr__(self):
        return "Nil"


@dataclass
class SymbolLiteral(ASTNode):
    """Symbol literal (:symbol)."""
    name: str
    
    def __repr__(self):
        return f"Symbol(:{self.name})"


@dataclass
class RegexLiteral(ASTNode):
    """Regex literal (/pattern/flags)."""
    pattern: str
    flags: str = ""
    
    def __repr__(self):
        return f"Regex(/{self.pattern}/{self.flags})"


@dataclass
class ListLiteral(ASTNode):
    """List/array literal."""
    elements: list
    
    def __repr__(self):
        return f"List({self.elements})"


@dataclass
class DictLiteral(ASTNode):
    """Dictionary/hash literal."""
    pairs: list  # List of (key, value) tuples
    
    def __repr__(self):
        return f"Dict({self.pairs})"


# ============================================================================
# Expressions
# ============================================================================

@dataclass
class Identifier(ASTNode):
    """Variable or function name."""
    name: str
    
    def __repr__(self):
        return f"Id({self.name})"


@dataclass
class BinaryOp(ASTNode):
    """Binary operation (e.g., a + b)."""
    left: ASTNode
    operator: str
    right: ASTNode
    
    def __repr__(self):
        return f"BinOp({self.left} {self.operator} {self.right})"


@dataclass
class UnaryOp(ASTNode):
    """Unary operation (e.g., -x, not x)."""
    operator: str
    operand: ASTNode
    
    def __repr__(self):
        return f"UnaryOp({self.operator} {self.operand})"


@dataclass
class Call(ASTNode):
    """Function call."""
    callee: ASTNode
    arguments: list
    
    def __repr__(self):
        return f"Call({self.callee}, args={self.arguments})"


@dataclass
class MethodCall(ASTNode):
    """Method call on an object (obj.method(args))."""
    object: ASTNode
    method: str
    arguments: list
    
    def __repr__(self):
        return f"MethodCall({self.object}.{self.method}({self.arguments}))"


@dataclass
class Index(ASTNode):
    """Index access (e.g., list[0])."""
    object: ASTNode
    index: ASTNode
    
    def __repr__(self):
        return f"Index({self.object}[{self.index}])"


@dataclass
class Slice(ASTNode):
    """Slice access (e.g., list[1:3])."""
    object: ASTNode
    start: Optional[ASTNode]
    end: Optional[ASTNode]
    step: Optional[ASTNode] = None
    
    def __repr__(self):
        return f"Slice({self.object}[{self.start}:{self.end}:{self.step}])"


@dataclass
class PropertyAccess(ASTNode):
    """Property access (e.g., obj.property)."""
    object: ASTNode
    property: str
    optional: bool = False  # For optional chaining (?.)
    
    def __repr__(self):
        op = "?." if self.optional else "."
        return f"Property({self.object}{op}{self.property})"


@dataclass
class Lambda(ASTNode):
    """Lambda/anonymous function."""
    params: list
    body: ASTNode
    
    def __repr__(self):
        return f"Lambda({self.params} -> {self.body})"


@dataclass
class PipeExpr(ASTNode):
    """Pipe expression (value |> function)."""
    value: ASTNode
    function: ASTNode
    
    def __repr__(self):
        return f"Pipe({self.value} |> {self.function})"


@dataclass
class TernaryExpr(ASTNode):
    """Ternary conditional (cond ? then : else)."""
    condition: ASTNode
    then_expr: ASTNode
    else_expr: ASTNode
    
    def __repr__(self):
        return f"Ternary({self.condition} ? {self.then_expr} : {self.else_expr})"


@dataclass
class ListComprehension(ASTNode):
    """List comprehension [expr for var in iterable if condition]."""
    expr: ASTNode
    var: str
    iterable: ASTNode
    condition: Optional[ASTNode] = None
    
    def __repr__(self):
        return f"ListComp({self.expr} for {self.var} in {self.iterable})"


@dataclass
class SpreadExpr(ASTNode):
    """Spread operator (...expr)."""
    expr: ASTNode
    
    def __repr__(self):
        return f"Spread(...{self.expr})"


# ============================================================================
# Statements
# ============================================================================

@dataclass
class LetStatement(ASTNode):
    """Variable declaration (let x = value)."""
    name: str
    value: ASTNode
    mutable: bool = False
    
    def __repr__(self):
        mut = "mut " if self.mutable else ""
        return f"Let({mut}{self.name} = {self.value})"


@dataclass
class Assignment(ASTNode):
    """Variable assignment (x = value)."""
    target: ASTNode
    value: ASTNode
    
    def __repr__(self):
        return f"Assign({self.target} = {self.value})"


@dataclass
class IfExpr(ASTNode):
    """If expression/statement."""
    condition: ASTNode
    then_branch: ASTNode
    else_branch: Optional[ASTNode] = None
    is_unless: bool = False
    
    def __repr__(self):
        keyword = "unless" if self.is_unless else "if"
        return f"If({keyword} {self.condition} then {self.then_branch} else {self.else_branch})"


@dataclass
class WhileLoop(ASTNode):
    """While loop."""
    condition: ASTNode
    body: ASTNode
    is_until: bool = False
    
    def __repr__(self):
        keyword = "until" if self.is_until else "while"
        return f"While({keyword} {self.condition} do {self.body})"


@dataclass
class ForLoop(ASTNode):
    """For-in loop."""
    var: str
    iterable: ASTNode
    body: ASTNode
    
    def __repr__(self):
        return f"For({self.var} in {self.iterable} do {self.body})"


@dataclass
class MatchCase(ASTNode):
    """A single case in a match expression."""
    pattern: ASTNode
    guard: Optional[ASTNode]
    body: ASTNode
    
    def __repr__(self):
        guard_str = f" if {self.guard}" if self.guard else ""
        return f"Case({self.pattern}{guard_str} -> {self.body})"


@dataclass
class MatchExpr(ASTNode):
    """Pattern matching expression."""
    value: ASTNode
    cases: list  # List of MatchCase
    
    def __repr__(self):
        return f"Match({self.value}, cases={self.cases})"


@dataclass
class Wildcard(ASTNode):
    """Wildcard pattern (_)."""
    
    def __repr__(self):
        return "Wildcard(_)"


@dataclass
class FunctionDef(ASTNode):
    """Function definition."""
    name: str
    params: list
    body: ASTNode
    
    def __repr__(self):
        return f"FnDef({self.name}({self.params}) = {self.body})"


@dataclass
class ReturnStatement(ASTNode):
    """Return statement."""
    value: Optional[ASTNode]
    
    def __repr__(self):
        return f"Return({self.value})"


@dataclass
class BreakStatement(ASTNode):
    """Break statement."""
    
    def __repr__(self):
        return "Break"


@dataclass
class ContinueStatement(ASTNode):
    """Continue statement."""
    
    def __repr__(self):
        return "Continue"


@dataclass
class Block(ASTNode):
    """Block of statements."""
    statements: list
    
    def __repr__(self):
        return f"Block({self.statements})"


@dataclass
class Program(ASTNode):
    """Top-level program."""
    statements: list
    
    def __repr__(self):
        return f"Program({self.statements})"


@dataclass
class BlockWithParams(ASTNode):
    """Block with parameters (Ruby-style do |x, y| ... end)."""
    params: list
    body: ASTNode
    
    def __repr__(self):
        return f"BlockWithParams(|{self.params}| {self.body})"


@dataclass
class PostfixIf(ASTNode):
    """Postfix conditional (expr if condition)."""
    expr: ASTNode
    condition: ASTNode
    is_unless: bool = False
    
    def __repr__(self):
        keyword = "unless" if self.is_unless else "if"
        return f"PostfixIf({self.expr} {keyword} {self.condition})"
