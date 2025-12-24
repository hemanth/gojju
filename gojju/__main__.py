"""
Command-line interface for the Gojju programming language.
"""

import sys
import argparse
from pathlib import Path

from . import __version__
from .lexer import Lexer, LexerError
from .parser import Parser, ParseError
from .interpreter import Interpreter, GojjuRuntimeError
from .repl import start_repl


def run_file(filepath: str, interpreter: Interpreter = None) -> int:
    """Execute a Gojju source file."""
    try:
        source = Path(filepath).read_text()
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        return 1
    
    return run_source(source, interpreter, filepath)


def run_source(source: str, interpreter: Interpreter = None, filename: str = "<stdin>") -> int:
    """Execute Gojju source code."""
    if interpreter is None:
        interpreter = Interpreter()
    
    try:
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        result = interpreter.execute(ast)
        
        # Only print result for -e flag (inline code)
        if filename == "<inline>":
            if result is not None:
                print(result)
        
        return 0
    
    except LexerError as e:
        print(f"{filename}:{e.line}:{e.column}: Lexer Error: {e.message}", file=sys.stderr)
        return 1
    
    except ParseError as e:
        print(f"{filename}:{e.token.line}:{e.token.column}: Parse Error: {e.message}", file=sys.stderr)
        return 1
    
    except GojjuRuntimeError as e:
        print(f"{filename}: Runtime Error: {e}", file=sys.stderr)
        return 1
    
    except Exception as e:
        print(f"{filename}: Error: {e}", file=sys.stderr)
        return 1


def main():
    """Main entry point for the Gojju CLI."""
    parser = argparse.ArgumentParser(
        prog='gojju',
        description='Gojju programming language - the essence of Python, Ruby, Haskell, Perl, and JavaScript',
        epilog='Examples:\n'
               '  gojju                    Start the REPL\n'
               '  gojju script.gj          Run a script\n'
               '  gojju -e "print 42"      Execute code\n',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'file',
        nargs='?',
        help='Gojju script file to execute (.gj)'
    )
    
    parser.add_argument(
        '-e', '--eval',
        metavar='CODE',
        help='Evaluate and execute code'
    )
    
    parser.add_argument(
        '-c', '--check',
        action='store_true',
        help='Check syntax without executing'
    )
    
    parser.add_argument(
        '--tokens',
        action='store_true',
        help='Print tokens (for debugging)'
    )
    
    parser.add_argument(
        '--ast',
        action='store_true',
        help='Print AST (for debugging)'
    )
    
    parser.add_argument(
        '-v', '--version',
        action='version',
        version=f'Gojju {__version__}'
    )
    
    args = parser.parse_args()
    
    # Handle inline code execution
    if args.eval:
        if args.tokens or args.ast or args.check:
            return debug_source(args.eval, args.tokens, args.ast, args.check)
        return run_source(args.eval, filename="<inline>")
    
    # Handle file execution
    if args.file:
        if args.tokens or args.ast or args.check:
            try:
                source = Path(args.file).read_text()
                return debug_source(source, args.tokens, args.ast, args.check, args.file)
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                return 1
        return run_file(args.file)
    
    # Start REPL if no arguments
    start_repl()
    return 0


def debug_source(source: str, show_tokens: bool, show_ast: bool, check_only: bool, filename: str = "<stdin>") -> int:
    """Debug/analyze source code."""
    try:
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        
        if show_tokens:
            print("=== TOKENS ===")
            for token in tokens:
                print(f"  {token}")
            print()
        
        parser = Parser(tokens)
        ast = parser.parse()
        
        if show_ast:
            print("=== AST ===")
            print_ast(ast)
            print()
        
        if check_only:
            print(f"✓ {filename}: Syntax OK")
        
        return 0
    
    except LexerError as e:
        print(f"{filename}:{e.line}:{e.column}: Lexer Error: {e.message}", file=sys.stderr)
        return 1
    
    except ParseError as e:
        print(f"{filename}:{e.token.line}:{e.token.column}: Parse Error: {e.message}", file=sys.stderr)
        return 1


def print_ast(node, indent=0):
    """Pretty-print an AST node."""
    prefix = "  " * indent
    
    if hasattr(node, 'statements'):
        print(f"{prefix}{type(node).__name__}:")
        for stmt in node.statements:
            print_ast(stmt, indent + 1)
    elif hasattr(node, '__dict__'):
        attrs = {k: v for k, v in node.__dict__.items() if not k.startswith('_')}
        if attrs:
            print(f"{prefix}{type(node).__name__}:")
            for key, value in attrs.items():
                if hasattr(value, '__dict__') and not isinstance(value, str):
                    print(f"{prefix}  {key}:")
                    print_ast(value, indent + 2)
                elif isinstance(value, list):
                    print(f"{prefix}  {key}:")
                    for item in value:
                        if hasattr(item, '__dict__'):
                            print_ast(item, indent + 2)
                        else:
                            print(f"{prefix}    {item}")
                else:
                    print(f"{prefix}  {key}: {value}")
        else:
            print(f"{prefix}{type(node).__name__}")
    else:
        print(f"{prefix}{node}")


if __name__ == '__main__':
    sys.exit(main())
