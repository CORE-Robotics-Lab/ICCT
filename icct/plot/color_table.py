"""
Compute \cellcolor[HTML]{...} hex codes for LaTeX table cells.
Global normalization across ALL methods and ALL environments in the table.

Two separate gradients (both orange=bad, white=good):
  - Reward cells:  orange = low reward,  white = high reward
  - Param cells:   orange = many params, white = few params

Usage:
    python icct/plot/color_table.py
"""

ENVS = ['ip', 'lunar', 'lane_keeping', 'ring_accel', 'ring_lc', 'figure8']

# Reward values: [ip, lunar, lane_keeping, ring_accel, ring_lc, figure8]
reward_data = {
    'Linear Tree':           [737.9,   223.5,    49.1,   121.5,  1147.4,   916.1],
    'ICCT-1-feature':        [1000.0,  190.1,   437.6,   121.6,  1269.6,  1072.4],
    'ICCT-2-feature':        [1000.0,  258.4,   458.5,   121.9,  1280.4,  1088.6],
    'ICCT-3-feature':        [1000.0,  275.8,   448.8,   120.8,  1280.8,  1048.7],
    'ICCT-L1-sparse':        [1000.0,  265.2,   465.5,   121.5,  1275.3,   993.2],
    'ICCT-complete':         [1000.0,  300.5,   476.6,   120.7,  1248.6,   994.1],
    'CDDT-ctrl-Crisp':       [  84.0, -126.6, -39826.4,   97.9,   639.6,   245.5],
    'MLP-Lower':             [1000.0,  231.6,   474.7,   121.8,   646.4,   868.4],
    'MLP-Upper':             [1000.0,  288.7,   467.9,   121.8,  1239.5,  1077.7],
    'MLP-Max':               [1000.0,  298.5,   478.2,   121.7,  1011.9,  1104.3],
    # Add CDDT row when available
    # 'CDDT':                [?,       ?,        ?,        ?,       ?,        ?],
}

# Param counts: [ip, lunar, lane_keeping, ring_accel, ring_lc, figure8]
param_data = {
    'Linear Tree':           [    54,    158,     91,      45,     950,     246],
    'ICCT-1-feature':        [    45,     69,     93,      93,     141,      93],
    'ICCT-2-feature':        [    29,    101,    125,     125,     205,     125],
    'ICCT-3-feature':        [    17,    133,    157,     157,     269,     157],
    'ICCT-L1-sparse':        [    29,    165,    253,     765,    2189,     509],
    'ICCT-complete':         [    13,    165,    253,     765,    2189,     509],
    'CDDT-ctrl-Crisp':       [    13,    165,    253,     765,    2189,     509],
    'MLP-Lower':             [    79,    110,    127,     151,     221,     103],
    'MLP-Upper':             [   121,    222,    407,     709,    3266,    1021],
    'MLP-Max':               [ 67329,  68610,  69377,   77569,   83458,   73473],
}


def value_to_hex(t):
    t = max(0.0, min(1.0, t))
    r = 255
    g = int(round(127 + t * 128))
    b = int(round(t * 255))
    return f'{r:02X}{g:02X}{b:02X}'


def global_range(data):
    all_vals = [v[i] for v in data.values() for i in range(len(ENVS)) if v[i] is not None]
    return min(all_vals), max(all_vals)


def main():
    r_min, r_max = global_range(reward_data)
    p_min, p_max = global_range(param_data)
    print(f'% Reward global range: [{r_min}, {r_max}]')
    print(f'% Param  global range: [{p_min}, {p_max}]')
    print()

    for method in reward_data:
        rvals = reward_data[method]
        pvals = param_data[method]

        rcells, pcells = [], []
        for i in range(len(ENVS)):
            rv, pv = rvals[i], pvals[i]
            rt = (rv - r_min) / (r_max - r_min)
            pt = 1.0 - (pv - p_min) / (p_max - p_min)
            rcells.append(f'\\cellcolor[HTML]{{{value_to_hex(rt)}}}')
            pcells.append(f'\\cellcolor[HTML]{{{value_to_hex(pt)}}}')

        print(f'% --- {method} ---')
        print('% Reward: ' + ' & '.join(rcells) + r' \\')
        print('% Params: ' + ' & '.join(pcells) + r' \\')
        print()


if __name__ == '__main__':
    main()
