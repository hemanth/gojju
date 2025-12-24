"""
Interactive REPL for the Gojju programming language.
"""

import sys
import os
import readline
import atexit
from typing import Optional

from .lexer import Lexer, LexerError
from .parser import Parser, ParseError
from .interpreter import Interpreter, GojjuRuntimeError


# ANSI color codes
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    BG_RED = '\033[41m'
    
    @classmethod
    def disable(cls):
        """Disable colors (for non-terminal output)."""
        for attr in dir(cls):
            if not attr.startswith('_') and attr.isupper():
                setattr(cls, attr, '')


def colorize(text: str, *colors) -> str:
    """Apply colors to text."""
    if not colors:
        return text
    return ''.join(colors) + text + Colors.RESET


BANNER = f"""
{Colors.CYAN}{Colors.BOLD}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ██████╗  ██████╗      ██╗     ██╗██╗   ██╗                 ║
║  ██╔════╝ ██╔═══██╗     ██║     ██║██║   ██║                 ║
║  ██║  ███╗██║   ██║     ██║     ██║██║   ██║                 ║
║  ██║   ██║██║   ██║██   ██║██   ██║██║   ██║                 ║
║  ╚██████╔╝╚██████╔╝╚█████╔╝╚█████╔╝╚██████╔╝                 ║
║   ╚═════╝  ╚═════╝  ╚════╝  ╚════╝  ╚═════╝                  ║
║                                                              ║
║  The essence of Python • Ruby • Haskell • Perl • JavaScript  ║
╚══════════════════════════════════════════════════════════════╝{Colors.RESET}

{Colors.DIM}Version 0.1.0 | Type .help for commands | .exit to quit{Colors.RESET}
"""

HELP_TEXT = f"""
{Colors.BOLD}Commands:{Colors.RESET}
  {Colors.CYAN}.help{Colors.RESET}     Show this help message
  {Colors.CYAN}.exit{Colors.RESET}     Exit the REPL
  {Colors.CYAN}.quit{Colors.RESET}     Exit the REPL
  {Colors.CYAN}.clear{Colors.RESET}    Clear the screen
  {Colors.CYAN}.env{Colors.RESET}      Show current environment
  {Colors.CYAN}.reset{Colors.RESET}    Reset the interpreter
  {Colors.CYAN}.load{Colors.RESET}     Load and execute a file

{Colors.BOLD}Syntax Examples:{Colors.RESET}
  {Colors.GREEN}# Variables{Colors.RESET}
  let x = 42
  mut counter = 0
  
  {Colors.GREEN}# Functions{Colors.RESET}
  fn double(n)
    n * 2
  end
  
  {Colors.GREEN}# Lambdas (arrow & Haskell-style){Colors.RESET}
  let add = (a, b) => a + b
  let inc = \\x -> x + 1
  
  {Colors.GREEN}# Pipe operator{Colors.RESET}
  [1, 2, 3] |> map(\\x -> x * 2) |> sum
  
  {Colors.GREEN}# Pattern matching{Colors.RESET}
  match x
    0 -> "zero"
    n if n < 0 -> "negative"
    _ -> "positive"
  end
  
  {Colors.GREEN}# List comprehensions{Colors.RESET}
  [x * x for x in range(10) if x % 2 == 0]
  
  {Colors.GREEN}# Functional programming{Colors.RESET}
  let composed = compose(double, inc)
  let curried = curry(add)
  curried(1)(2)
"""


class REPL:
    """Interactive REPL for Gojju."""
    
    def __init__(self):
        self.interpreter = Interpreter()
        self.history_file = os.path.expanduser('~/.gojju_history')
        self._setup_readline()
        self.multiline_buffer = []
        self.multiline_mode = False
    
    def _setup_readline(self):
        """Set up readline for history and editing."""
        try:
            readline.read_history_file(self.history_file)
        except FileNotFoundError:
            pass
        
        readline.set_history_length(1000)
        atexit.register(self._save_history)
        
        # Set up tab completion
        readline.parse_and_bind('tab: complete')
        readline.set_completer(self._completer)
    
    def _save_history(self):
        """Save history to file."""
        try:
            readline.write_history_file(self.history_file)
        except:
            pass
    
    def _completer(self, text: str, state: int) -> Optional[str]:
        """Tab completion."""
        options = [name for name in self.interpreter.env.keys() if name.startswith(text)]
        
        # Add keywords
        keywords = ['let', 'mut', 'fn', 'if', 'else', 'unless', 'while', 'until', 
                    'for', 'in', 'do', 'end', 'return', 'match', 'true', 'false', 'nil']
        options.extend(k for k in keywords if k.startswith(text))
        
        if state < len(options):
            return options[state]
        return None
    
    def _needs_continuation(self, code: str) -> bool:
        """Check if code needs more input (unbalanced brackets, etc.)."""
        # Count brackets
        opens = code.count('(') + code.count('[') + code.count('{')
        closes = code.count(')') + code.count(']') + code.count('}')
        
        if opens > closes:
            return True
        
        # Check for keywords that need 'end'
        lines = code.split('\n')
        keywords_needing_end = ['fn', 'if', 'unless', 'while', 'until', 'for', 'match', 'do']
        
        depth = 0
        for line in lines:
            stripped = line.strip()
            words = stripped.split()
            if words:
                if words[0] in keywords_needing_end:
                    depth += 1
                elif words[0] == 'end':
                    depth -= 1
        
        return depth > 0
    
    def _get_prompt(self) -> str:
        """Get the appropriate prompt."""
        if self.multiline_mode:
            return colorize('... ', Colors.DIM)
        return colorize('gojju> ', Colors.CYAN, Colors.BOLD)
    
    def _format_result(self, result) -> str:
        """Format a result for display."""
        if result is None:
            return colorize('nil', Colors.DIM)
        if isinstance(result, bool):
            return colorize(str(result).lower(), Colors.MAGENTA)
        if isinstance(result, (int, float)):
            return colorize(str(result), Colors.YELLOW)
        if isinstance(result, str):
            return colorize(f'"{result}"', Colors.GREEN)
        if isinstance(result, list):
            items = ', '.join(self._format_result(x) for x in result)
            return f'[{items}]'
        if isinstance(result, dict):
            items = ', '.join(f'{self._format_result(k)}: {self._format_result(v)}' 
                            for k, v in result.items())
            return f'{{{items}}}'
        return colorize(str(result), Colors.WHITE)
    
    def _handle_command(self, cmd: str) -> bool:
        """Handle REPL commands (starting with .). Returns True to continue, False to exit."""
        cmd = cmd.strip().lower()
        
        if cmd in ('.exit', '.quit', '.q'):
            print(colorize('\nGoodbye! 👋', Colors.CYAN))
            return False
        
        if cmd == '.help':
            print(HELP_TEXT)
            return True
        
        if cmd == '.clear':
            os.system('clear' if os.name != 'nt' else 'cls')
            return True
        
        if cmd == '.env':
            print(colorize('\nCurrent Environment:', Colors.BOLD))
            for name, value in sorted(self.interpreter.env.items()):
                if not name.startswith('_'):
                    print(f'  {colorize(name, Colors.CYAN)}: {self._format_result(value)}')
            print()
            return True
        
        if cmd == '.reset':
            self.interpreter = Interpreter()
            print(colorize('Interpreter reset.', Colors.GREEN))
            return True
        
        if cmd.startswith('.load '):
            filename = cmd[6:].strip()
            return self._load_file(filename)
        
        print(colorize(f"Unknown command: {cmd}", Colors.RED))
        return True
    
    def _load_file(self, filename: str) -> bool:
        """Load and execute a file."""
        try:
            with open(filename, 'r') as f:
                source = f.read()
            
            lexer = Lexer(source)
            tokens = lexer.tokenize()
            parser = Parser(tokens)
            ast = parser.parse()
            result = self.interpreter.execute(ast)
            
            print(colorize(f"Loaded: {filename}", Colors.GREEN))
            if result is not None:
                print(f'=> {self._format_result(result)}')
        except FileNotFoundError:
            print(colorize(f"File not found: {filename}", Colors.RED))
        except Exception as e:
            print(colorize(f"Error loading {filename}: {e}", Colors.RED))
        
        return True
    
    def _execute(self, source: str) -> None:
        """Execute code and display result."""
        try:
            lexer = Lexer(source)
            tokens = lexer.tokenize()
            parser = Parser(tokens)
            ast = parser.parse()
            result = self.interpreter.execute(ast)
            
            if result is not None:
                print(f'=> {self._format_result(result)}')
        
        except LexerError as e:
            print(colorize(f'Lexer Error: {e.message}', Colors.RED))
            print(colorize(f'  at line {e.line}, column {e.column}', Colors.DIM))
        
        except ParseError as e:
            print(colorize(f'Parse Error: {e.message}', Colors.RED))
            print(colorize(f'  at line {e.token.line}, column {e.token.column}', Colors.DIM))
        
        except GojjuRuntimeError as e:
            print(colorize(f'Runtime Error: {e}', Colors.RED))
        
        except Exception as e:
            print(colorize(f'Error: {e}', Colors.RED))
    
    def run(self):
        """Run the REPL."""
        # Check if we're in a terminal
        if not sys.stdout.isatty():
            Colors.disable()
        
        print(BANNER)
        
        while True:
            try:
                line = input(self._get_prompt())
                
                # Handle empty input
                if not line.strip():
                    if self.multiline_mode:
                        # Empty line ends multiline input
                        source = '\n'.join(self.multiline_buffer)
                        self.multiline_buffer = []
                        self.multiline_mode = False
                        self._execute(source)
                    continue
                
                # Handle commands
                if line.strip().startswith('.') and not self.multiline_mode:
                    if not self._handle_command(line):
                        break
                    continue
                
                # Handle multiline input
                if self.multiline_mode:
                    self.multiline_buffer.append(line)
                    if not self._needs_continuation('\n'.join(self.multiline_buffer)):
                        source = '\n'.join(self.multiline_buffer)
                        self.multiline_buffer = []
                        self.multiline_mode = False
                        self._execute(source)
                    continue
                
                # Check if we need to continue to next line
                if self._needs_continuation(line):
                    self.multiline_mode = True
                    self.multiline_buffer = [line]
                    continue
                
                # Execute single line
                self._execute(line)
            
            except KeyboardInterrupt:
                print(colorize('\n(Use .exit to quit)', Colors.DIM))
                self.multiline_buffer = []
                self.multiline_mode = False
            
            except EOFError:
                print(colorize('\nGoodbye! 👋', Colors.CYAN))
                break


def start_repl():
    """Start the Gojju REPL."""
    repl = REPL()
    repl.run()


if __name__ == '__main__':
    start_repl()
