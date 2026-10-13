"""
Flagship Project 01: Pareto Curve Generator & Teardown Visualizer
Parses benchmark metrics and generates:
1. Terminal ASCII Pareto frontier tables
2. Multi-chart interactive HTML report (Throughput vs TTFT, Context Scaling, MFU & FinOps breakdown)
"""

import json
import os
import sys

def generate_html_chart(results: list, output_html: str = "pareto_curve.html"):
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Frontier LLM Inference Benchmark Teardown | NVIDIA A100 Pareto Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: #0b0f19;
            color: #f1f5f9;
            padding: 30px;
            margin: 0;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{ margin-bottom: 30px; border-bottom: 1px solid #1e293b; padding-bottom: 20px; }}
        h1 {{ color: #38bdf8; margin: 0 0 8px 0; font-size: 28px; }}
        .subtitle {{ color: #94a3b8; font-size: 15px; }}
        .grid-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 25px;
        }}
        .stat-card {{
            background: #131c2e;
            padding: 16px 20px;
            border-radius: 10px;
            border: 1px solid #1e293b;
        }}
        .stat-label {{ color: #64748b; font-size: 12px; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; }}
        .stat-val {{ font-size: 24px; font-weight: 700; color: #38bdf8; margin-top: 4px; }}
        .stat-sub {{ font-size: 12px; color: #10b981; margin-top: 2px; }}
        
        .card {{
            background: #131c2e;
            padding: 24px;
            border-radius: 12px;
            border: 1px solid #1e293b;
            margin-bottom: 30px;
        }}
        .card h2 {{ color: #e2e8f0; font-size: 18px; margin-top: 0; margin-bottom: 15px; }}
        
        .charts-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }}
        @media (max-width: 900px) {{
            .charts-grid {{ grid-template-columns: 1fr; }}
        }}

        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 13px; }}
        th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #1e293b; }}
        th {{ color: #38bdf8; background: #0b0f19; font-weight: 600; position: sticky; top: 0; }}
        tr:hover {{ background: #1a233a; }}
        .badge {{ background: #0284c7; color: white; padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 500; }}
        .badge-green {{ background: #059669; color: white; padding: 2px 8px; border-radius: 6px; font-size: 11px; }}
    </style>
</head>
<body>
<div class="container">
    <header>
        <h1>Frontier LLM Inference Benchmark Teardown</h1>
        <div class="subtitle">Hardware: NVIDIA A100-SXM4-40GB | Workload: Meta-Llama-3-8B | Sweep: Concurrency 1–256 & Contexts 512–4,096 tokens</div>
    </header>

    <div class="grid-stats">
        <div class="stat-card">
            <div class="stat-label">Peak Serving Throughput</div>
            <div class="stat-val" id="statPeakTok">--</div>
            <div class="stat-sub">SGLang (RadixAttention)</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Lowest P99 TTFT</div>
            <div class="stat-val" id="statLowestTTFT">--</div>
            <div class="stat-sub">Prefix-Cached Hit</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Lowest FinOps Cost Floor</div>
            <div class="stat-val" id="statLowestCost">--</div>
            <div class="stat-sub">Per 1M Output Tokens</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Peak Model FLOPs (MFU)</div>
            <div class="stat-val" id="statPeakMFU">--</div>
            <div class="stat-sub">A100 Tensor Core Saturation</div>
        </div>
    </div>

    <div class="card">
        <h2>Pareto Frontier: Aggregate Throughput vs P99 TTFT (Bubble Size = FinOps Efficiency)</h2>
        <canvas id="paretoChart" height="110"></canvas>
    </div>

    <div class="charts-grid">
        <div class="card">
            <h2>Throughput Scaling vs Concurrency</h2>
            <canvas id="concurrencyChart" height="160"></canvas>
        </div>
        <div class="card">
            <h2>Cost per 1M Tokens ($) vs Concurrency</h2>
            <canvas id="costChart" height="160"></canvas>
        </div>
    </div>

    <div class="card">
        <h2>Empirical Benchmark Data Points (All Context Windows & Engines)</h2>
        <div style="max-height: 400px; overflow-y: auto;">
            <table>
                <thead>
                    <tr>
                        <th>Engine</th>
                        <th>Context</th>
                        <th>Concurrency</th>
                        <th>Throughput (tok/s)</th>
                        <th>P99 TTFT</th>
                        <th>P99 ITL</th>
                        <th>MFU (%)</th>
                        <th>Cost / 1M Tok</th>
                    </tr>
                </thead>
                <tbody>
"""
    for r in results:
        mfu_str = f"{r.get('mfu_percent', 0):.1f}%"
        html_content += f"""                    <tr>
                        <td><b>{r['engine']}</b></td>
                        <td><span class="badge-green">{r['prompt_len']} tok</span></td>
                        <td><span class="badge">{r['concurrency']}</span></td>
                        <td>{r['throughput_tok_s']:.1f}</td>
                        <td>{r['ttft_p99_ms']:.1f} ms</td>
                        <td>{r['itl_p99_ms']:.2f} ms</td>
                        <td>{mfu_str}</td>
                        <td>${r['cost_per_1m_tokens_usd']:.4f}</td>
                    </tr>\n"""

    html_content += """                </tbody>
            </table>
        </div>
    </div>
</div>

<script>
const data = """ + json.dumps(results) + """;

// Populate Header Stats
const maxTok = Math.max(...data.map(d => d.throughput_tok_s));
const minTTFT = Math.min(...data.map(d => d.ttft_p99_ms));
const minCost = Math.min(...data.map(d => d.cost_per_1m_tokens_usd));
const maxMFU = Math.max(...data.map(d => d.mfu_percent || 0));

document.getElementById('statPeakTok').innerText = maxTok.toFixed(1) + ' tok/s';
document.getElementById('statLowestTTFT').innerText = minTTFT.toFixed(1) + ' ms';
document.getElementById('statLowestCost').innerText = '$' + minCost.toFixed(4);
document.getElementById('statPeakMFU').innerText = maxMFU.toFixed(1) + '%';

const engines = [...new Set(data.map(d => d.engine))];
const colors = ['#38bdf8', '#34d399', '#f43f5e'];

// 1. Pareto Chart
const ctxPareto = document.getElementById('paretoChart').getContext('2d');
const paretoDatasets = engines.map((eng, idx) => ({
    label: eng,
    data: data.filter(d => d.engine === eng).map(d => ({
        x: d.ttft_p99_ms,
        y: d.throughput_tok_s,
        r: Math.max(5, Math.min(18, 32 / Math.sqrt(d.cost_per_1m_tokens_usd + 0.05)))
    })),
    backgroundColor: colors[idx % colors.length]
}));

new Chart(ctxPareto, {
    type: 'bubble',
    data: { datasets: paretoDatasets },
    options: {
        responsive: true,
        plugins: {
            legend: { labels: { color: '#cbd5e1' } }
        },
        scales: {
            x: { title: { display: true, text: 'P99 Time to First Token (ms) [Lower is Better]', color: '#94a3b8' }, ticks: { color: '#94a3b8' }, grid: { color: '#1e293b' } },
            y: { title: { display: true, text: 'Serving Throughput (tok/s) [Higher is Better]', color: '#94a3b8' }, ticks: { color: '#94a3b8' }, grid: { color: '#1e293b' } }
        }
    }
});

// 2. Concurrency Scaling Chart (Aggregated over 512 context)
const ctxConc = document.getElementById('concurrencyChart').getContext('2d');
const concData = data.filter(d => d.prompt_len === 512);
const concurrencies = [...new Set(concData.map(d => d.concurrency))].sort((a,b) => a-b);

const concDatasets = engines.map((eng, idx) => ({
    label: eng,
    data: concurrencies.map(c => {
        const pt = concData.find(d => d.engine === eng && d.concurrency === c);
        return pt ? pt.throughput_tok_s : null;
    }),
    borderColor: colors[idx % colors.length],
    backgroundColor: colors[idx % colors.length],
    tension: 0.3
}));

new Chart(ctxConc, {
    type: 'line',
    data: { labels: concurrencies.map(c => 'Conc ' + c), datasets: concDatasets },
    options: {
        responsive: true,
        plugins: { legend: { labels: { color: '#cbd5e1' } } },
        scales: {
            x: { ticks: { color: '#94a3b8' }, grid: { color: '#1e293b' } },
            y: { title: { display: true, text: 'Throughput (tok/s)', color: '#94a3b8' }, ticks: { color: '#94a3b8' }, grid: { color: '#1e293b' } }
        }
    }
});

// 3. Cost Chart
const ctxCost = document.getElementById('costChart').getContext('2d');
const costDatasets = engines.map((eng, idx) => ({
    label: eng,
    data: concurrencies.map(c => {
        const pt = concData.find(d => d.engine === eng && d.concurrency === c);
        return pt ? pt.cost_per_1m_tokens_usd : null;
    }),
    borderColor: colors[idx % colors.length],
    backgroundColor: colors[idx % colors.length],
    tension: 0.3
}));

new Chart(ctxCost, {
    type: 'line',
    data: { labels: concurrencies.map(c => 'Conc ' + c), datasets: costDatasets },
    options: {
        responsive: true,
        plugins: { legend: { labels: { color: '#cbd5e1' } } },
        scales: {
            x: { ticks: { color: '#94a3b8' }, grid: { color: '#1e293b' } },
            y: { title: { display: true, text: 'Cost ($ / 1M Tokens)', color: '#94a3b8' }, ticks: { color: '#94a3b8' }, grid: { color: '#1e293b' } }
        }
    }
});
</script>
</body>
</html>"""
    with open(output_html, "w") as f:
        f.write(html_content)
    print(f"[+] Interactive Pareto dashboard generated: '{output_html}'")

def main():
    json_path = "benchmark_results.json"
    if not os.path.exists(json_path):
        print(f"Error: '{json_path}' not found. Run 'python run_benchmarks.py' first!")
        sys.exit(1)

    with open(json_path) as f:
        results = json.load(f)

    print("\n" + "=" * 95)
    print(" EMPIRICAL PARETO FRONTIER TABLE")
    print("=" * 95)
    print(f"{'Engine':<32} | {'Prompt':<6} | {'Conc':<5} | {'Throughput':<12} | {'P99 TTFT':<10} | {'Cost / 1M Tok'}")
    print("-" * 95)
    for r in results:
        print(f"{r['engine'][:32]:<32} | {r.get('prompt_len', 512):<6} | {r['concurrency']:<5} | {r['throughput_tok_s']:>7.1f} tok/s | {r['ttft_p99_ms']:>6.1f} ms | ${r['cost_per_1m_tokens_usd']:>6.4f}")
    print("=" * 95)

    generate_html_chart(results, "pareto_curve.html")

if __name__ == "__main__":
    main()
