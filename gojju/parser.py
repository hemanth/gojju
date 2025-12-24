"""
Parser for the Gojju programming language.
Converts a stream of tokens into an Abstract Syntax Tree (AST).
"""

from typing import List, Optional, Callable
from .lexer import Token, TokenType, LexerError
from .ast import *


class ParseError(Exception):
    """Parser error with position information."""
    def __init__(self, message: str, token: Token):
        self.message = message
        self.token = token
        super().__init__(f"Line {token.line}, Column {token.column}: {message}")


class Parser:
    """Recursive descent parser for Gojju."""
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
    
    @property
    def current(self) -> Token:
        """Current token."""
        if self.pos >= len(self.tokens):
            return self.tokens[-1]  # Return EOF
        return self.tokens[self.pos]
    
    def peek(self, offset: int = 1) -> Token:
        """Peek ahead by offset tokens."""
        pos = self.pos + offset
        if pos >= len(self.tokens):
            return self.tokens[-1]
        return self.tokens[pos]
    
    def advance(self) -> Token:
        """Advance to next token and return current."""
        token = self.current
        if self.pos < len(self.tokens):
            self.pos += 1
        return token
    
    def match(self, *types: TokenType) -> bool:
        """Check if current token matches any of the types."""
        return self.current.type in types
    
    def consume(self, type: TokenType, message: str = None) -> Token:
        """Consume a token of the expected type."""
        if self.current.type == type:
            return self.advance()
        if message is None:
            message = f"Expected {type.name}, got {self.current.type.name}"
        raise ParseError(message, self.current)
    
    def skip_newlines(self):
        """Skip newline tokens."""
        while self.match(TokenType.NEWLINE):
            self.advance()
    
    def parse(self) -> Program:
        """Parse the entire program."""
        statements = []
        self.skip_newlines()
        
        while not self.match(TokenType.EOF):
            stmt = self.parse_statement()
            if stmt is not None:
                statements.append(stmt)
            self.skip_newlines()
        
        return Program(statements)
    
    def parse_statement(self) -> Optional[ASTNode]:
        """Parse a single statement."""
        self.skip_newlines()
        
        if self.match(TokenType.LET, TokenType.MUT):
            return self.parse_let_statement()
        
        if self.match(TokenType.FN):
            return self.parse_function_def()
        
        if self.match(TokenType.IF):
            return self.parse_if_expr()
        
        if self.match(TokenType.UNLESS):
            return self.parse_unless_expr()
        
        if self.match(TokenType.WHILE):
            return self.parse_while_loop()
        
        if self.match(TokenType.UNTIL):
            return self.parse_until_loop()
        
        if self.match(TokenType.FOR):
            return self.parse_for_loop()
        
        if self.match(TokenType.MATCH):
            return self.parse_match_expr()
        
        if self.match(TokenType.RETURN):
            return self.parse_return_statement()
        
        if self.match(TokenType.BREAK):
            self.advance()
            return BreakStatement()
        
        if self.match(TokenType.CONTINUE):
            self.advance()
            return ContinueStatement()
        
        # Expression statement
        expr = self.parse_expression()
        
        # Check for postfix if/unless
        if self.match(TokenType.IF):
            self.advance()
            condition = self.parse_expression()
            return PostfixIf(expr, condition, is_unless=False)
        
        if self.match(TokenType.UNLESS):
            self.advance()
            condition = self.parse_expression()
            return PostfixIf(expr, condition, is_unless=True)
        
        # Check for assignment
        if self.match(TokenType.ASSIGN):
            self.advance()
            value = self.parse_expression()
            return Assignment(expr, value)
        
        if self.match(TokenType.PLUS_ASSIGN, TokenType.MINUS_ASSIGN, 
                      TokenType.STAR_ASSIGN, TokenType.SLASH_ASSIGN):
            op_token = self.advance()
            value = self.parse_expression()
            # Convert += to a = a + value
            op = {
                TokenType.PLUS_ASSIGN: '+',
                TokenType.MINUS_ASSIGN: '-',
                TokenType.STAR_ASSIGN: '*',
                TokenType.SLASH_ASSIGN: '/',
            }[op_token.type]
            return Assignment(expr, BinaryOp(expr, op, value))
        
        return expr
    
    def parse_let_statement(self) -> LetStatement:
        """Parse let/mut variable declaration."""
        is_mutable = self.current.type == TokenType.MUT
        self.advance()  # Skip let/mut
        
        if self.current.type == TokenType.MUT:
            is_mutable = True
            self.advance()
        
        name_token = self.consume(TokenType.IDENTIFIER, "Expected variable name")
        self.consume(TokenType.ASSIGN, "Expected '=' in variable declaration")
        value = self.parse_expression()
        
        return LetStatement(name_token.value, value, mutable=is_mutable)
    
    def parse_function_def(self) -> FunctionDef:
        """Parse function definition."""
        self.advance()  # Skip 'fn'
        
        name_token = self.consume(TokenType.IDENTIFIER, "Expected function name")
        name = name_token.value
        
        # Parse parameters
        self.consume(TokenType.LPAREN, "Expected '(' after function name")
        params = []
        
        while not self.match(TokenType.RPAREN):
            if params:
                self.consume(TokenType.COMMA, "Expected ',' between parameters")
            param_token = self.consume(TokenType.IDENTIFIER, "Expected parameter name")
            params.append(param_token.value)
        
        self.consume(TokenType.RPAREN, "Expected ')' after parameters")
        self.skip_newlines()
        
        # Parse body (until 'end')
        body = self.parse_block_body()
        
        return FunctionDef(name, params, body)
    
    def parse_block_body(self) -> Block:
        """Parse statements until 'end'."""
        statements = []
        self.skip_newlines()
        
        while not self.match(TokenType.END, TokenType.ELSE, TokenType.EOF):
            stmt = self.parse_statement()
            if stmt is not None:
                statements.append(stmt)
            self.skip_newlines()
        
        if self.match(TokenType.END):
            self.advance()
        
        return Block(statements)
    
    def parse_if_expr(self) -> IfExpr:
        """Parse if expression."""
        self.advance()  # Skip 'if'
        
        condition = self.parse_expression()
        self.skip_newlines()
        
        # Check for optional 'then' or just newline
        if self.match(TokenType.IDENTIFIER) and self.current.value == 'then':
            self.advance()
        
        then_branch = self.parse_if_body()
        
        else_branch = None
        if self.match(TokenType.ELSE):
            self.advance()
            self.skip_newlines()
            
            if self.match(TokenType.IF):
                else_branch = self.parse_if_expr()
            else:
                else_branch = self.parse_block_body()
        elif self.match(TokenType.END):
            self.advance()
        
        return IfExpr(condition, then_branch, else_branch)
    
    def parse_if_body(self) -> Block:
        """Parse if body until else or end."""
        statements = []
        self.skip_newlines()
        
        while not self.match(TokenType.END, TokenType.ELSE, TokenType.EOF):
            stmt = self.parse_statement()
            if stmt is not None:
                statements.append(stmt)
            self.skip_newlines()
        
        return Block(statements)
    
    def parse_unless_expr(self) -> IfExpr:
        """Parse unless expression (negated if)."""
        self.advance()  # Skip 'unless'
        
        condition = self.parse_expression()
        self.skip_newlines()
        
        then_branch = self.parse_block_body()
        
        return IfExpr(condition, then_branch, is_unless=True)
    
    def parse_while_loop(self) -> WhileLoop:
        """Parse while loop."""
        self.advance()  # Skip 'while'
        
        condition = self.parse_expression()
        self.skip_newlines()
        
        # Optional 'do'
        if self.match(TokenType.DO):
            self.advance()
        
        body = self.parse_block_body()
        
        return WhileLoop(condition, body)
    
    def parse_until_loop(self) -> WhileLoop:
        """Parse until loop (negated while)."""
        self.advance()  # Skip 'until'
        
        condition = self.parse_expression()
        self.skip_newlines()
        
        if self.match(TokenType.DO):
            self.advance()
        
        body = self.parse_block_body()
        
        return WhileLoop(condition, body, is_until=True)
    
    def parse_for_loop(self) -> ForLoop:
        """Parse for-in loop."""
        self.advance()  # Skip 'for'
        
        var_token = self.consume(TokenType.IDENTIFIER, "Expected loop variable")
        self.consume(TokenType.IN, "Expected 'in' in for loop")
        iterable = self.parse_expression()
        self.skip_newlines()
        
        if self.match(TokenType.DO):
            self.advance()
        
        body = self.parse_block_body()
        
        return ForLoop(var_token.value, iterable, body)
    
    def parse_match_expr(self) -> MatchExpr:
        """Parse match expression."""
        self.advance()  # Skip 'match'
        
        value = self.parse_expression()
        self.skip_newlines()
        
        cases = []
        while not self.match(TokenType.END, TokenType.EOF):
            case = self.parse_match_case()
            if case:
                cases.append(case)
            self.skip_newlines()
        
        self.consume(TokenType.END, "Expected 'end' after match")
        
        return MatchExpr(value, cases)
    
    def parse_match_case(self) -> Optional[MatchCase]:
        """Parse a single match case."""
        self.skip_newlines()
        
        if self.match(TokenType.END, TokenType.EOF):
            return None
        
        # Parse pattern
        if self.match(TokenType.IDENTIFIER) and self.current.value == '_':
            self.advance()
            pattern = Wildcard()
        else:
            pattern = self.parse_primary()
        
        # Optional guard
        guard = None
        if self.match(TokenType.IF):
            self.advance()
            guard = self.parse_expression()
        
        self.consume(TokenType.ARROW, "Expected '->' in match case")
        
        # Parse body
        body = self.parse_expression()
        
        return MatchCase(pattern, guard, body)
    
    def parse_return_statement(self) -> ReturnStatement:
        """Parse return statement."""
        self.advance()  # Skip 'return'
        
        value = None
        if not self.match(TokenType.NEWLINE, TokenType.EOF, TokenType.END):
            value = self.parse_expression()
        
        return ReturnStatement(value)
    
    def parse_expression(self) -> ASTNode:
        """Parse an expression (entry point)."""
        return self.parse_pipe()
    
    def parse_pipe(self) -> ASTNode:
        """Parse pipe expressions (|>)."""
        left = self.parse_or()
        
        while self.match(TokenType.PIPE):
            self.advance()
            right = self.parse_or()
            left = PipeExpr(left, right)
        
        return left
    
    def parse_or(self) -> ASTNode:
        """Parse or expressions."""
        left = self.parse_and()
        
        while self.match(TokenType.OR):
            self.advance()
            right = self.parse_and()
            left = BinaryOp(left, 'or', right)
        
        return left
    
    def parse_and(self) -> ASTNode:
        """Parse and expressions."""
        left = self.parse_equality()
        
        while self.match(TokenType.AND):
            self.advance()
            right = self.parse_equality()
            left = BinaryOp(left, 'and', right)
        
        return left
    
    def parse_equality(self) -> ASTNode:
        """Parse equality expressions (==, !=)."""
        left = self.parse_comparison()
        
        while self.match(TokenType.EQ, TokenType.NE):
            op = self.advance()
            right = self.parse_comparison()
            left = BinaryOp(left, op.value, right)
        
        return left
    
    def parse_comparison(self) -> ASTNode:
        """Parse comparison expressions (<, >, <=, >=)."""
        left = self.parse_range()
        
        while self.match(TokenType.LT, TokenType.GT, TokenType.LE, TokenType.GE):
            op = self.advance()
            right = self.parse_range()
            left = BinaryOp(left, op.value, right)
        
        return left
    
    def parse_range(self) -> ASTNode:
        """Parse range expressions (..)."""
        left = self.parse_term()
        
        if self.match(TokenType.RANGE):
            self.advance()
            right = self.parse_term()
            return BinaryOp(left, '..', right)
        
        return left
    
    def parse_term(self) -> ASTNode:
        """Parse additive expressions (+, -)."""
        left = self.parse_factor()
        
        while self.match(TokenType.PLUS, TokenType.MINUS):
            op = self.advance()
            right = self.parse_factor()
            left = BinaryOp(left, op.value, right)
        
        return left
    
    def parse_factor(self) -> ASTNode:
        """Parse multiplicative expressions (*, /, %)."""
        left = self.parse_power()
        
        while self.match(TokenType.STAR, TokenType.SLASH, TokenType.PERCENT):
            op = self.advance()
            right = self.parse_power()
            left = BinaryOp(left, op.value, right)
        
        return left
    
    def parse_power(self) -> ASTNode:
        """Parse power expressions (**)."""
        left = self.parse_unary()
        
        if self.match(TokenType.POWER):
            self.advance()
            right = self.parse_power()  # Right associative
            left = BinaryOp(left, '**', right)
        
        return left
    
    def parse_unary(self) -> ASTNode:
        """Parse unary expressions (-, not)."""
        if self.match(TokenType.MINUS):
            self.advance()
            return UnaryOp('-', self.parse_unary())
        
        if self.match(TokenType.NOT):
            self.advance()
            return UnaryOp('not', self.parse_unary())
        
        return self.parse_postfix()
    
    def parse_postfix(self) -> ASTNode:
        """Parse postfix expressions (calls, indexing, property access)."""
        expr = self.parse_primary()
        
        while True:
            if self.match(TokenType.LPAREN):
                # Function call
                self.advance()
                args = []
                
                while not self.match(TokenType.RPAREN):
                    if args:
                        self.consume(TokenType.COMMA, "Expected ',' between arguments")
                    args.append(self.parse_expression())
                
                self.consume(TokenType.RPAREN, "Expected ')' after arguments")
                expr = Call(expr, args)
            
            elif self.match(TokenType.LBRACKET):
                # Indexing or slicing
                self.advance()
                
                start = None
                if not self.match(TokenType.COLON):
                    start = self.parse_expression()
                
                if self.match(TokenType.COLON):
                    # Slice
                    self.advance()
                    end = None
                    if not self.match(TokenType.RBRACKET, TokenType.COLON):
                        end = self.parse_expression()
                    
                    step = None
                    if self.match(TokenType.COLON):
                        self.advance()
                        step = self.parse_expression()
                    
                    self.consume(TokenType.RBRACKET, "Expected ']'")
                    expr = Slice(expr, start, end, step)
                else:
                    # Index
                    self.consume(TokenType.RBRACKET, "Expected ']'")
                    expr = Index(expr, start)
            
            elif self.match(TokenType.DOT):
                # Property access
                self.advance()
                prop = self.consume(TokenType.IDENTIFIER, "Expected property name")
                
                # Check for method call
                if self.match(TokenType.LPAREN):
                    self.advance()
                    args = []
                    
                    while not self.match(TokenType.RPAREN):
                        if args:
                            self.consume(TokenType.COMMA, "Expected ',' between arguments")
                        args.append(self.parse_expression())
                    
                    self.consume(TokenType.RPAREN, "Expected ')'")
                    expr = MethodCall(expr, prop.value, args)
                else:
                    expr = PropertyAccess(expr, prop.value)
            
            elif self.match(TokenType.OPTIONAL_DOT):
                # Optional chaining
                self.advance()
                prop = self.consume(TokenType.IDENTIFIER, "Expected property name")
                expr = PropertyAccess(expr, prop.value, optional=True)
            
            elif self.match(TokenType.DO):
                # Block passed to method
                self.advance()
                block = self.parse_block_with_params()
                
                # Treat as method call with block as last argument
                if isinstance(expr, MethodCall):
                    expr.arguments.append(block)
                elif isinstance(expr, Call):
                    expr.arguments.append(block)
            
            else:
                break
        
        return expr
    
    def parse_block_with_params(self) -> BlockWithParams:
        """Parse a block with parameters (do |x, y| ... end)."""
        params = []
        
        if self.match(TokenType.PIPE_CHAR):
            self.advance()
            while not self.match(TokenType.PIPE_CHAR):
                if params:
                    self.consume(TokenType.COMMA, "Expected ',' between block parameters")
                param = self.consume(TokenType.IDENTIFIER, "Expected block parameter")
                params.append(param.value)
            self.consume(TokenType.PIPE_CHAR, "Expected '|'")
        
        body = self.parse_block_body()
        
        return BlockWithParams(params, body)
    
    def parse_primary(self) -> ASTNode:
        """Parse primary expressions (literals, identifiers, grouped expressions)."""
        # Numbers
        if self.match(TokenType.NUMBER):
            token = self.advance()
            return NumberLiteral(token.value)
        
        # Strings
        if self.match(TokenType.STRING):
            token = self.advance()
            return StringLiteral(token.value)
        
        # Booleans
        if self.match(TokenType.TRUE):
            self.advance()
            return BooleanLiteral(True)
        
        if self.match(TokenType.FALSE):
            self.advance()
            return BooleanLiteral(False)
        
        # Nil
        if self.match(TokenType.NIL):
            self.advance()
            return NilLiteral()
        
        # Symbols
        if self.match(TokenType.SYMBOL):
            token = self.advance()
            return SymbolLiteral(token.value)
        
        # Regex
        if self.match(TokenType.REGEX):
            token = self.advance()
            pattern, flags = token.value
            return RegexLiteral(pattern, flags)
        
        # Identifiers
        if self.match(TokenType.IDENTIFIER):
            token = self.advance()
            if token.value == '_':
                return Wildcard()
            return Identifier(token.value)
        
        # Grouped expression or lambda
        if self.match(TokenType.LPAREN):
            self.advance()
            
            # Check for arrow function: (params) => body
            # We need to look ahead to see if this is a lambda
            if self.match(TokenType.RPAREN):
                self.advance()
                if self.match(TokenType.FAT_ARROW):
                    # Empty parameter list lambda
                    self.advance()
                    body = self.parse_expression()
                    return Lambda([], body)
                else:
                    # Empty tuple/unit? Or just ()
                    return NilLiteral()
            
            # Try to parse as parameters list for lambda
            first = self.parse_expression()
            
            if self.match(TokenType.COMMA):
                # Could be lambda params
                params = [first]
                while self.match(TokenType.COMMA):
                    self.advance()
                    params.append(self.parse_expression())
                
                self.consume(TokenType.RPAREN, "Expected ')'")
                
                if self.match(TokenType.FAT_ARROW):
                    self.advance()
                    # Extract identifiers from params
                    param_names = []
                    for p in params:
                        if isinstance(p, Identifier):
                            param_names.append(p.name)
                        else:
                            raise ParseError("Lambda parameters must be identifiers", self.current)
                    body = self.parse_expression()
                    return Lambda(param_names, body)
                else:
                    # It was a tuple (not supported yet, treat as grouped)
                    raise ParseError("Unexpected comma", self.current)
            
            self.consume(TokenType.RPAREN, "Expected ')'")
            
            if self.match(TokenType.FAT_ARROW):
                # Single parameter lambda
                self.advance()
                if isinstance(first, Identifier):
                    body = self.parse_expression()
                    return Lambda([first.name], body)
                else:
                    raise ParseError("Lambda parameter must be identifier", self.current)
            
            return first
        
        # List literal or list comprehension
        if self.match(TokenType.LBRACKET):
            return self.parse_list_or_comprehension()
        
        # Dict literal
        if self.match(TokenType.LBRACE):
            return self.parse_dict_literal()
        
        # Haskell-style lambda: \x -> x + 1
        if self.match(TokenType.LAMBDA):
            return self.parse_haskell_lambda()
        
        # Spread operator
        if self.match(TokenType.SPREAD):
            self.advance()
            expr = self.parse_unary()
            return SpreadExpr(expr)
        
        raise ParseError(f"Unexpected token: {self.current.type.name}", self.current)
    
    def parse_list_or_comprehension(self) -> ASTNode:
        """Parse list literal or list comprehension."""
        self.advance()  # Skip '['
        
        if self.match(TokenType.RBRACKET):
            self.advance()
            return ListLiteral([])
        
        first = self.parse_expression()
        
        # Check for list comprehension
        if self.match(TokenType.FOR):
            self.advance()
            var_token = self.consume(TokenType.IDENTIFIER, "Expected variable in comprehension")
            self.consume(TokenType.IN, "Expected 'in' in comprehension")
            iterable = self.parse_expression()
            
            condition = None
            if self.match(TokenType.IF):
                self.advance()
                condition = self.parse_expression()
            
            self.consume(TokenType.RBRACKET, "Expected ']'")
            return ListComprehension(first, var_token.value, iterable, condition)
        
        # Regular list
        elements = [first]
        while self.match(TokenType.COMMA):
            self.advance()
            if self.match(TokenType.RBRACKET):
                break
            elements.append(self.parse_expression())
        
        self.consume(TokenType.RBRACKET, "Expected ']'")
        return ListLiteral(elements)
    
    def parse_dict_literal(self) -> DictLiteral:
        """Parse dictionary literal."""
        self.advance()  # Skip '{'
        
        pairs = []
        
        while not self.match(TokenType.RBRACE):
            if pairs:
                self.consume(TokenType.COMMA, "Expected ',' between dict entries")
            
            if self.match(TokenType.RBRACE):
                break
            
            key = self.parse_expression()
            self.consume(TokenType.COLON, "Expected ':' in dict entry")
            value = self.parse_expression()
            pairs.append((key, value))
        
        self.consume(TokenType.RBRACE, "Expected '}'")
        return DictLiteral(pairs)
    
    def parse_haskell_lambda(self) -> Lambda:
        """Parse Haskell-style lambda: \\x -> x + 1"""
        self.advance()  # Skip '\\'
        
        params = []
        while self.match(TokenType.IDENTIFIER):
            params.append(self.advance().value)
        
        self.consume(TokenType.ARROW, "Expected '->' in lambda")
        body = self.parse_expression()
        
        return Lambda(params, body)
