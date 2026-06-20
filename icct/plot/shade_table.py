"""
Apply \cellcolor shading to LaTeX table files.

Reads a .tex file, finds numeric cells, and replaces them with
\cellcolor[HTML]{...}{value} using a gradient from orange (bad) to white (good).

Two columns are shaded independently:
  - Reward columns: higher = better (white), lower = worse (orange)
  - Param columns:  fewer = better (white), more = worse (orange)

Scale options:
  linear      — uniform gradient
  exponential — compresses the good end, expands differences at the bad end.
                Use --exp-base to control steepness (default 5).

Usage:
    python icct/plot/shade_table.py input.tex output_shaded.tex \
        --reward-cols 1 3 5 7 9 11 \
        --param-cols  2 4 6 8 10 12 \
        --scale exponential --exp-base 5

Column indices are 1-based, matching the LaTeX table column order.
"""

import argparse
import re
import sys


def value_to_hex(t: float) -> str:
    """Map t in [0,1] (1=good/white, 0=bad/orange) to an HTML hex color."""
    t = max(0.0, min(1.0, t))
    r = 255
    g = int(round(127 + t * 128))
    b = int(round(t * 255))
    return f'{r:02X}{g:02X}{b:02X}'


def normalize_linear(values: list, higher_is_better: bool) -> list:
    valid = [v for v in values if v is not None]
    if not valid or max(valid) == min(valid):
        return [1.0 if v is not None else None for v in values]
    lo, hi = min(valid), max(valid)
    result = []
    for v in values:
        if v is None:
            result.append(None)
        else:
            t = (v - lo) / (hi - lo)
            result.append(t if higher_is_better else 1.0 - t)
    return result


def normalize_exponential(values: list, higher_is_better: bool, base: float) -> list:
    """Exponential normalization: spreads out differences at the bad end."""
    linear = normalize_linear(values, higher_is_better)
    result = []
    for t in linear:
        if t is None:
            result.append(None)
        else:
            # base^t - 1) / (base - 1) maps [0,1]->[0,1] with exponential weighting
            t_exp = (base ** t - 1.0) / (base - 1.0)
            result.append(t_exp)
    return result


def shade_cells(row_cells: list, col_indices: set, t_map: dict) -> list:
    """Insert \\cellcolor into the cells at the given 1-based col indices."""
    result = []
    for i, cell in enumerate(row_cells):
        col = i + 1
        if col in col_indices and col in t_map and t_map[col] is not None:
            hex_color = value_to_hex(t_map[col])
            stripped = cell.strip()
            result.append(f'\\cellcolor[HTML]{{{hex_color}}}{{{stripped}}}')
        else:
            result.append(cell)
    return result


def extract_number(s: str):
    """Try to parse a float from a LaTeX cell string. Returns None on failure."""
    s = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', s)  # strip latex commands
    s = re.sub(r'[{}]', '', s).strip()
    try:
        return float(s.replace('\\', '').strip())
    except ValueError:
        return None


def process_table(lines: list, reward_cols: set, param_cols: set,
                  scale: str, exp_base: float) -> list:
    # First pass: collect all numeric values per column
    col_values = {}
    data_rows = []

    for line in lines:
        if '&' not in line:
            data_rows.append(None)
            continue
        cells = line.rstrip('\n').rstrip('\\\\').split('&')
        row_nums = {}
        for i, cell in enumerate(cells):
            col = i + 1
            if col in reward_cols or col in param_cols:
                v = extract_number(cell)
                row_nums[col] = v
                col_values.setdefault(col, []).append(v)
        data_rows.append((cells, row_nums))

    # Compute normalized t values per column
    col_t = {}
    for col, vals in col_values.items():
        higher_is_better = col in reward_cols
        if scale == 'exponential':
            col_t[col] = normalize_exponential(vals, higher_is_better, exp_base)
        else:
            col_t[col] = normalize_linear(vals, higher_is_better)

    # Second pass: emit shaded lines
    row_idx = {col: 0 for col in col_values}
    output = []
    for line, row_data in zip(lines, data_rows):
        if row_data is None:
            output.append(line)
            continue
        cells, _ = row_data
        t_map = {}
        for col in col_values:
            if row_idx[col] < len(col_t[col]):
                t_map[col] = col_t[col][row_idx[col]]
                row_idx[col] += 1
        all_cols = reward_cols | param_cols
        shaded = shade_cells(cells, all_cols, t_map)
        suffix = ' \\\\\n' if line.rstrip('\n').endswith('\\\\') else '\n'
        output.append(' & '.join(shaded) + suffix)

    return output


def main():
    parser = argparse.ArgumentParser(description='Shade LaTeX table cells by value')
    parser.add_argument('input', help='Input .tex file')
    parser.add_argument('output', help='Output shaded .tex file')
    parser.add_argument('--reward-cols', nargs='+', type=int, default=[],
                        metavar='COL', help='1-based column indices for reward cells')
    parser.add_argument('--param-cols', nargs='+', type=int, default=[],
                        metavar='COL', help='1-based column indices for param cells')
    parser.add_argument('--scale', choices=['linear', 'exponential'], default='exponential')
    parser.add_argument('--exp-base', type=float, default=5.0,
                        help='Base for exponential scale (default: 5)')
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        lines = f.readlines()

    reward_cols = set(args.reward_cols)
    param_cols = set(args.param_cols)

    if not reward_cols and not param_cols:
        print("Warning: no --reward-cols or --param-cols specified; output unchanged.",
              file=sys.stderr)

    shaded = process_table(lines, reward_cols, param_cols, args.scale, args.exp_base)

    with open(args.output, 'w') as f:
        f.writelines(shaded)

    print(f"Shaded table written to {args.output}")


if __name__ == '__main__':
    main()
