using PandaSharp.Column;

namespace PandaSharp.Expressions;

/// <summary>
/// Recursive descent parser for string expressions.
/// Converts "price * quantity > 1000" into an Expr tree.
///
/// Grammar:
///   expr     → or_expr
///   or_expr  → and_expr ("or" and_expr)*
///   and_expr → not_expr ("and" not_expr)*
///   not_expr → "not" not_expr | compare
///   compare  → additive ((">" | "<" | ">=" | "<=" | "==" | "!=") additive)?
///   additive → mult (("+" | "-") mult)*
///   mult     → unary (("*" | "/") unary)*
///   unary    → "-" unary | primary
///   primary  → NUMBER | STRING | IDENT | "(" expr ")"
/// </summary>
public static class ExprParser
{
    /// <summary>
    /// Parse an expression string into an Expr tree.
    /// Column names are bare identifiers: "price * quantity > 1000"
    /// String literals use single quotes: "name == 'Alice'"
    /// </summary>
    public static Expr Parse(string expression)
    {
        var tokens = Tokenize(expression);
        int pos = 0;
        var result = ParseOrExpr(tokens, ref pos);
        if (pos < tokens.Count)
            throw new FormatException($"Unexpected token at position {pos}: '{tokens[pos].Value}'");
        return result;
    }

    // ===== Tokenizer =====

    private enum TokenType { Number, String, Ident, Op, LParen, RParen }
    private record Token(TokenType Type, string Value);

    private static List<Token> Tokenize(string expr)
    {
        var tokens = new List<Token>();
        int i = 0;
        while (i < expr.Length)
        {
            char c = expr[i];
            if (char.IsWhiteSpace(c)) { i++; continue; }

            // Number
            if (char.IsDigit(c) || (c == '.' && i + 1 < expr.Length && char.IsDigit(expr[i + 1])))
            {
                int start = i;
                while (i < expr.Length && (char.IsDigit(expr[i]) || expr[i] == '.')) i++;
                tokens.Add(new Token(TokenType.Number, expr[start..i]));
                continue;
            }

            // String literal (single-quoted)
            if (c == '\'')
            {
                i++;
                int start = i;
                while (i < expr.Length && expr[i] != '\'') i++;
                tokens.Add(new Token(TokenType.String, expr[start..i]));
                if (i < expr.Length) i++; // skip closing quote
                continue;
            }

            // Identifier or keyword
            if (char.IsLetter(c) || c == '_')
            {
                int start = i;
                while (i < expr.Length && (char.IsLetterOrDigit(expr[i]) || expr[i] == '_')) i++;
                var word = expr[start..i];
                tokens.Add(new Token(TokenType.Ident, word));
                continue;
            }

            // Operators
            if (c == '(' ) { tokens.Add(new Token(TokenType.LParen, "(")); i++; continue; }
            if (c == ')') { tokens.Add(new Token(TokenType.RParen, ")")); i++; continue; }

            // Two-char operators
            if (i + 1 < expr.Length)
            {
                string two = expr.Substring(i, 2);
                if (two is ">=" or "<=" or "==" or "!=")
                {
                    tokens.Add(new Token(TokenType.Op, two));
                    i += 2;
                    continue;
                }
            }

            // Single-char operators
            if (c is '+' or '-' or '*' or '/' or '>' or '<')
            {
                tokens.Add(new Token(TokenType.Op, c.ToString()));
                i++;
                continue;
            }

            throw new FormatException($"Unexpected character '{c}' at position {i}");
        }
        return tokens;
    }

    // ===== Parser =====

    private static Expr ParseOrExpr(List<Token> tokens, ref int pos)
    {
        var left = ParseAndExpr(tokens, ref pos);
        while (pos < tokens.Count && tokens[pos] is { Type: TokenType.Ident, Value: "or" })
        {
            pos++;
            var right = ParseAndExpr(tokens, ref pos);
            left = left | right;
        }
        return left;
    }

    private static Expr ParseAndExpr(List<Token> tokens, ref int pos)
    {
        var left = ParseNotExpr(tokens, ref pos);
        while (pos < tokens.Count && tokens[pos] is { Type: TokenType.Ident, Value: "and" })
        {
            pos++;
            var right = ParseNotExpr(tokens, ref pos);
            left = left & right;
        }
        return left;
    }

    private static Expr ParseNotExpr(List<Token> tokens, ref int pos)
    {
        if (pos < tokens.Count && tokens[pos] is { Type: TokenType.Ident, Value: "not" })
        {
            pos++;
            var operand = ParseNotExpr(tokens, ref pos);
            return !operand;
        }
        return ParseCompare(tokens, ref pos);
    }

    private static Expr ParseCompare(List<Token> tokens, ref int pos)
    {
        var left = ParseAdditive(tokens, ref pos);
        if (pos < tokens.Count && tokens[pos].Type == TokenType.Op)
        {
            var op = tokens[pos].Value;
            if (op is ">" or "<" or ">=" or "<=" or "==" or "!=")
            {
                pos++;
                var right = ParseAdditive(tokens, ref pos);
                return op switch
                {
                    ">" => left > right,
                    "<" => left < right,
                    ">=" => left >= right,
                    "<=" => left <= right,
                    "==" => left.Eq(right),
                    "!=" => left.Neq(right),
                    _ => throw new FormatException($"Unknown comparison operator: {op}")
                };
            }
        }
        return left;
    }

    private static Expr ParseAdditive(List<Token> tokens, ref int pos)
    {
        var left = ParseMultiplicative(tokens, ref pos);
        while (pos < tokens.Count && tokens[pos] is { Type: TokenType.Op, Value: "+" or "-" })
        {
            var op = tokens[pos].Value;
            pos++;
            var right = ParseMultiplicative(tokens, ref pos);
            left = op == "+" ? left + right : left - right;
        }
        return left;
    }

    private static Expr ParseMultiplicative(List<Token> tokens, ref int pos)
    {
        var left = ParseUnary(tokens, ref pos);
        while (pos < tokens.Count && tokens[pos] is { Type: TokenType.Op, Value: "*" or "/" })
        {
            var op = tokens[pos].Value;
            pos++;
            var right = ParseUnary(tokens, ref pos);
            left = op == "*" ? left * right : left / right;
        }
        return left;
    }

    private static Expr ParseUnary(List<Token> tokens, ref int pos)
    {
        if (pos < tokens.Count && tokens[pos] is { Type: TokenType.Op, Value: "-" })
        {
            pos++;
            var operand = ParseUnary(tokens, ref pos);
            return Expr.Lit(0) - operand;
        }
        return ParsePrimary(tokens, ref pos);
    }

    private static Expr ParsePrimary(List<Token> tokens, ref int pos)
    {
        if (pos >= tokens.Count)
            throw new FormatException("Unexpected end of expression.");

        var token = tokens[pos];

        // Number literal
        if (token.Type == TokenType.Number)
        {
            pos++;
            if (token.Value.Contains('.'))
                return Expr.Lit(double.Parse(token.Value, System.Globalization.CultureInfo.InvariantCulture));
            else
                return Expr.Lit(int.Parse(token.Value));
        }

        // String literal
        if (token.Type == TokenType.String)
        {
            pos++;
            return Expr.Lit(token.Value);
        }

        // Parenthesized expression
        if (token.Type == TokenType.LParen)
        {
            pos++;
            var inner = ParseOrExpr(tokens, ref pos);
            if (pos >= tokens.Count || tokens[pos].Type != TokenType.RParen)
                throw new FormatException("Missing closing parenthesis.");
            pos++;
            return inner;
        }

        // Identifier → column reference (skip keywords)
        if (token.Type == TokenType.Ident && token.Value is not "and" and not "or" and not "not")
        {
            pos++;
            return Expr.Col(token.Value);
        }

        throw new FormatException($"Unexpected token: '{token.Value}'");
    }
}

/// <summary>Extension methods for string-based Eval on DataFrames.</summary>
public static class EvalExtensions
{
    /// <summary>
    /// Evaluate a string expression against this DataFrame.
    /// Returns a filtered DataFrame if the expression is boolean,
    /// or a DataFrame with a computed column otherwise.
    ///
    /// Usage:
    ///   df.Eval("price * quantity > 1000")    // filter rows
    ///   df.Eval("total = price * quantity")    // add computed column
    /// </summary>
    public static DataFrame Eval(this DataFrame df, string expression)
    {
        // Check for assignment: "name = expr"
        int eqIdx = expression.IndexOf('=');
        if (eqIdx > 0 && expression[eqIdx - 1] != '!' && expression[eqIdx - 1] != '<' &&
            expression[eqIdx - 1] != '>' &&
            (eqIdx + 1 >= expression.Length || expression[eqIdx + 1] != '='))
        {
            var colName = expression[..eqIdx].Trim();
            var exprStr = expression[(eqIdx + 1)..].Trim();

            // Validate column name (simple identifier)
            if (colName.All(c => char.IsLetterOrDigit(c) || c == '_'))
            {
                var expr = ExprParser.Parse(exprStr);
                return df.WithColumn(expr, colName);
            }
        }

        // Parse as expression
        var parsed = ExprParser.Parse(expression);
        var result = parsed.Evaluate(df);

        // If result is boolean, filter; otherwise add as column
        if (result is Column<bool> boolCol)
        {
            var mask = new bool[df.RowCount];
            for (int i = 0; i < df.RowCount; i++)
                mask[i] = boolCol[i] ?? false;
            return df.Filter(mask);
        }

        // Return original df with computed column added
        return df.AddColumn(result);
    }

    /// <summary>
    /// Evaluate a string expression and return the result as a column.
    /// </summary>
    public static IColumn EvalColumn(this DataFrame df, string expression)
    {
        var expr = ExprParser.Parse(expression);
        return expr.Evaluate(df);
    }
}
