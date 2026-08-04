import re
import sympy
from sympy import symbols, diff, solve, sympify, simplify, latex

class MathSolver:
    def __init__(self):
        self.x = symbols('x')
        self.superscripts = {
            '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
            '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'
        }

    def normalize(self, expr: str) -> str:
        """
        Convert human-readable math into SymPy-friendly syntax.
        Handles: x², 3x, (x+1)(x-1), 2(x), etc.
        """
        # 1. Replace Unicode superscripts
        for char, val in self.superscripts.items():
            expr = expr.replace(char, f"**{val}")
        
        # 2. Standardize basic operators
        expr = expr.replace("÷", "/").replace("×", "*").replace("^", "**")
        
        # 3. Handle implicit multiplication between number and variable (e.g., 3x -> 3*x)
        expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)
        
        # 4. Handle implicit multiplication between variables (e.g., x y -> x*y)
        # We handle this specifically for 'x' followed by another letter or vice versa
        expr = re.sub(r'([a-zA-Z])\s+([a-zA-Z])', r'\1*\2', expr)
        
        # 5. Handle implicit multiplication with parentheses
        # Number followed by parenthesis: 3(x) -> 3*(x)
        expr = re.sub(r'(\d)\(', r'\1*(', expr)
        # Parenthesis followed by variable/number: (x)2 -> (x)*2, (x)y -> (x)*y
        expr = re.sub(r'\)([\d[a-zA-Z])', r')*\1', expr)
        # Variable followed by parenthesis: x(y) -> x*(y)
        # We must exclude known function names to avoid sin(x) -> sin*(x)
        functions = {"sin", "cos", "tan", "log", "ln", "exp", "sqrt", "asin", "acos", "atan"}
        def func_sub(match):
            prefix = match.group(1)
            if prefix.lower() in functions:
                return f"{prefix}("
            return f"{prefix}*("
        
        expr = re.sub(r'([a-zA-Z]+)\(', func_sub, expr)
        # Two parentheses: (x)(y) -> (x)*(y)
        expr = re.sub(r'\)\(', r')*(', expr)
        
        # 6. Cleanup whitespace
        expr = re.sub(r'\s+', ' ', expr).strip()
        return expr

    def solve_request(self, text: str):
        """
        Detects type of math operation and performs it.
        """
        # Remove leading "solve", "what is", etc. for better parsing
        clean_text = re.sub(r'^(?:solve|calculate|what is|find|evaluate|differentiate)\s+', '', text, flags=re.IGNORECASE).strip("? ")
        text_lower = text.lower()
        
        try:
            # 1. Detection: Differentiation
            if any(k in text_lower for k in ["derivative", "differentiate", "d/dx"]):
                # Extract expression after keywords
                expr_match = re.search(r'(?:derivative|differentiate|d/dx)\s+(?:of\s+)?(.+)', text_lower)
                raw_expr = expr_match.group(1).strip("? ") if expr_match else clean_text
                norm_expr = self.normalize(raw_expr)
                parsed = sympify(norm_expr)
                result = diff(parsed, self.x)
                return {
                    "type": "differentiation",
                    "original": raw_expr,
                    "normalized": norm_expr,
                    "result": str(result),
                    "latex_result": f"\\frac{{d}}{{dx}}({latex(parsed)}) = {latex(result)}",
                    "steps": [f"1. Identified expression: {raw_expr}", f"2. Normalized: {norm_expr}", f"3. Differentiated: {result}"]
                }

            # 2. Detection: Solving Equations (contains '=')
            if "=" in clean_text:
                parts = clean_text.split("=")
                left = self.normalize(parts[0])
                right = self.normalize(parts[1]) if len(parts) > 1 else "0"
                full_expr = f"({left}) - ({right})"
                parsed = sympify(full_expr)
                solutions = solve(parsed, self.x)
                
                # Format solutions as LaTeX
                lx_sols = ", ".join([latex(s) for s in solutions])
                return {
                    "type": "solving",
                    "original": text,
                    "result": [str(s) for s in solutions],
                    "latex_result": f"x = \\left\\{{ {lx_sols} \\right\\}}",
                    "steps": [f"1. Rearranged: {full_expr} = 0", f"2. Solved: {solutions}"]
                }

            # 3. Detection: Simplification / Basic Arithmetic
            norm_expr = self.normalize(clean_text)
            parsed = sympify(norm_expr)
            result = simplify(parsed)
            
            # Heuristic: only treat as math if it's not a common word and contains numbers or symbols
            if any(char.isdigit() or char in "+-*/^()" for char in norm_expr):
                return {
                    "type": "simplification",
                    "original": text,
                    "result": str(result),
                    "latex_result": f"{latex(parsed)} = {latex(result)}",
                    "steps": [f"1. Normalized: {norm_expr}", f"2. Evaluated: {result}"]
                }
            return None

        except Exception as e:
            print(f"[MathSolver] Error: {e}")
            return None

# Singleton
math_solver = MathSolver()

if __name__ == "__main__":
    # Quick CLI test
    test_cases = [
        "What is the derivative of x² + 3x?",
        "Solve x² - 5x + 6 = 0",
        "3x(x + 2)",
        "sin(x) + x⁴"
    ]
    for tc in test_cases:
        res = math_solver.solve_request(tc)
        print(f"Query: {tc} | Result: {res['result'] if res else 'FAILED'}")
