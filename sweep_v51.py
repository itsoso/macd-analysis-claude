#!/usr/bin/env python3
"""v5.1 参数扫描: 聚焦降低 MaxDD"""
import subprocess, sys, os, time, itertools, re, json

# 扫描参数
SHORT_SL = [-0.12, -0.14, -0.16]
SHORT_THRESHOLD = [40, 45, 50]
SHORT_TRAIL = [0.10, 0.12, 0.15]

results = []
combos = list(itertools.product(SHORT_SL, SHORT_THRESHOLD, SHORT_TRAIL))
print(f"=== v5.1 参数扫描: {len(combos)} 个组合 ===\n")

for i, (sl, st, trail) in enumerate(combos, 1):
    notes = f"sweep_v51: sl={sl} st={st} trail={trail}"
    cmd = [
        sys.executable, "backtest_multi_tf_daily.py",
        "--start", "2025-01-01", "--end", "2026-01-31",
        "--notes", notes,
        "--override", f"short_sl={sl}",
        "--override", f"short_threshold={st}",
        "--override", f"short_trail={trail}",
    ]
    print(f"[{i}/{len(combos)}] sl={sl} st={st} trail={trail} ...", end=" ", flush=True)
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = proc.stdout + proc.stderr
        elapsed = time.time() - t0

        # 解析输出
        def extract(pattern, text, default=0):
            m = re.search(pattern, text)
            return float(m.group(1)) if m else default

        run_id = extract(r'run_id=(\d+)', output, 0)
        ret = extract(r'策略收益:\s+([+\-\d.]+)%', output, 0)
        mdd = extract(r'最大回撤:\s+([+\-\d.]+)%', output, 0)
        ppf = extract(r'组合PF:\s+([\d.]+)', output, 0)
        cpf = extract(r'合约PF:\s+([\d.]+)', output, 0)
        wr = extract(r'胜率:\s+([\d.]+)%', output, 0)
        trades = extract(r'交易次数:\s+(\d+)', output, 0)

        r = {'sl': sl, 'st': st, 'trail': trail, 'run_id': int(run_id),
             'return': ret, 'maxdd': mdd, 'ppf': ppf, 'cpf': cpf,
             'trades': int(trades), 'winrate': wr}
        results.append(r)
        print(f"run#{r['run_id']} ret={r['return']:+.1f}% MDD={r['maxdd']:.1f}% pPF={r['ppf']:.2f} cPF={r['cpf']:.2f} ({elapsed:.0f}s)")
    except Exception as e:
        print(f"ERROR: {e}")

# 排序输出
print(f"\n{'='*80}")
print(f"=== 按 pPF 排序 (MaxDD >= -15%) ===")
print(f"{'='*80}")
filtered = [r for r in results if r['maxdd'] >= -15]
filtered.sort(key=lambda x: -x['ppf'])
print(f"{'rank':>4} {'run':>5} {'sl':>6} {'st':>4} {'trail':>6} {'return':>8} {'maxdd':>8} {'pPF':>6} {'cPF':>6} {'WR':>6}")
for i, r in enumerate(filtered[:15], 1):
    print(f"{i:4d} {r['run_id']:>5} {r['sl']:>6.2f} {r['st']:>4} {r['trail']:>6.2f} {r['return']:>+7.1f}% {r['maxdd']:>7.1f}% {r['ppf']:>6.2f} {r['cpf']:>6.2f} {r['winrate']:>5.1f}%")

print(f"\n{'='*80}")
print(f"=== 全部结果 (按 pPF 排序) ===")
print(f"{'='*80}")
results.sort(key=lambda x: -x['ppf'])
for i, r in enumerate(results, 1):
    flag = " ★" if r['maxdd'] >= -12 and r['ppf'] >= 1.9 else ""
    print(f"{i:4d} run#{r['run_id']:>3} sl={r['sl']:>6.2f} st={r['st']:>3} trail={r['trail']:>5.2f} | ret={r['return']:>+7.1f}% MDD={r['maxdd']:>7.1f}% pPF={r['ppf']:>5.2f} cPF={r['cpf']:>5.2f} WR={r['winrate']:>5.1f}%{flag}")
