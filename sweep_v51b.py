#!/usr/bin/env python3
"""v5.1 参数扫描 - 第二轮: 探索更宽止损区间 + 更高trail"""
import subprocess, sys, os, time, itertools, re

# 扫描参数 - 探索更宽的止损和更高的trail
SHORT_SL = [-0.16, -0.18, -0.20, -0.25]
SHORT_TRAIL = [0.15, 0.18, 0.20, 0.25]
SHORT_THRESHOLD = [40, 45]

results = []
combos = list(itertools.product(SHORT_SL, SHORT_THRESHOLD, SHORT_TRAIL))
print(f"=== v5.1 扫描B: {len(combos)} 个组合 (宽止损+高trail) ===\n")

for i, (sl, st, trail) in enumerate(combos, 1):
    notes = f"sweep_v51b: sl={sl} st={st} trail={trail}"
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
        tag = " ★" if ret > 100 else " ✗" if ret < 0 else ""
        print(f"run#{r['run_id']} ret={r['return']:+.1f}% MDD={r['maxdd']:.1f}% pPF={r['ppf']:.2f} cPF={r['cpf']:.2f} ({elapsed:.0f}s){tag}")
    except Exception as e:
        print(f"ERROR: {e}")

print(f"\n{'='*90}")
print(f"=== 全部结果 (按 pPF 排序) ===")
print(f"{'='*90}")
results.sort(key=lambda x: -x['ppf'])
for i, r in enumerate(results, 1):
    flag = " ★" if r['return'] > 100 else ""
    print(f"{i:4d} run#{r['run_id']:>3} sl={r['sl']:>6.2f} st={r['st']:>3} trail={r['trail']:>5.2f} | ret={r['return']:>+7.1f}% MDD={r['maxdd']:>7.1f}% pPF={r['ppf']:>5.2f} cPF={r['cpf']:>5.2f} WR={r['winrate']:>5.1f}%{flag}")

# 统计正收益组合的trail分布
print(f"\n=== 正收益组合 trail 分布 ===")
for trail in sorted(set(r['trail'] for r in results)):
    pos = [r for r in results if r['trail'] == trail and r['return'] > 0]
    neg = [r for r in results if r['trail'] == trail and r['return'] <= 0]
    print(f"trail={trail:.2f}: {len(pos)} 正 / {len(neg)} 负")
