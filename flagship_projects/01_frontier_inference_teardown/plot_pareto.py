"""
Flagship Project 01: Pareto Curve Generator & Teardown Visualizer
Parses benchmark metrics and generates:
1. Terminal ASCII Pareto frontier tables
2. Standalone HTML interactive chart ready for blog/Twitter publication
"""

import json
import os
import sys

def generate_html_chart(results: list, output_html: str = "pareto_curve.html"):
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>AI Infra: Frontier Inference Pareto Teardown</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; padding: 30px; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        h1 {{ color: #38bdf8; margin-bottom: 5px; }}
        .subtitle {{ color: #94a3b8; margin-bottom: 25px; }}
        .card {{ background: #1e293b; padding: 20px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 25px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 14px; }}
        th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ color: #38bdf8; background: #0f172a; }}
        .badge {{ background: #0284c7; color: white; padding: 2px 8px; border-radius: 6px; font-size: 11px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Frontier LLM Inference Benchmark Teardown</h1>
    <div class="subtitle">Hardware: NVIDIA A100-SXM4-40GB | Pareto Frontier: Cost vs TTFT vs Throughput</div>

    <div class="card">
        <canvas id="paretoChart" width="900" height="420"></canvas>
    </div>

    <div class="card">
        <h3>Empirical Hardware Results</h3>
        <table>
            <thead>
                <tr>
                    <th>Engine</th>
                    <th>Concurrency</th>
                    <th>Throughput (tok/s)</th>
                    <th>P99 TTFT (ms)</th>
                    <th>P99 ITL (ms)</th>
                    <th>Cost / 1M Tokens</th>
                </tr>
            </thead>
            <tbody>
"""
    for r in results:
        html_content += f"""                <tr>
                    <td><b>{r['engine']}</b></td>
                    <td><span class="badge">{r['concurrency']}</span></td>
                    <td>{r['throughput_tok_s']:.1f}</td>
                    <td>{r['ttft_p99_ms']:.1f}</td>
                    <td>{r['itl_p99_ms']:.2f}</td>
                    <td>${r['cost_per_1m_tokens_usd']:.4f}</td>
                </tr>\n"""

    html_content += """            </tbody>
        </table>
    </div>
</div>
<script>
const data = """ + json.dumps(results) + """;
const ctx = document.getElementById('paretoChart').getContext('2d');
const engines = [...new Set(data.map(d => d.engine))];
const colors = ['#38bdf8', '#34d399', '#f43f5e'];

const datasets = engines.map((eng, idx) => {
    const points = data.filter(d => d.engine === eng).map(d => ({
        x: d.ttft_p99_ms,
        y: d.throughput_tok_s,
        r: Math.max(5, Math.min(18, 30 / Math.sqrt(d.cost_per_1m_tokens_usd + 0.1)))
    }));
    return {
        label: eng,
        data: points,
        backgroundColor: colors[idx % colors.length]
    };
});

new Chart(ctx, {
    type: 'bubble',
    data: { datasets: datasets },
    options: {
        responsive: true,
        plugins: {
            title: { display: true, text: 'Pareto Frontier: Throughput vs P99 TTFT (Bubble Size = FinOps Efficiency)', color: '#f8fafc' },
            legend: { labels: { color: '#cbd5e1' } }
        },
        scales: {
            x: { title: { display: true, text: 'P99 Time to First Token (ms) [Lower is Better]', color: '#94a3b8' }, ticks: { color: '#94a3b8' }, grid: { color: '#334155' } },
            y: { title: { display: true, text: 'Aggregate Throughput (Tokens/sec) [Higher is Better]', color: '#94a3b8' }, ticks: { color: '#94a3b8' }, grid: { color: '#334155' } }
        }
    }
});
</script>
</body>
</html>"""
    with open(output_html, "w") as f:
        f.write(html_content)
    print(f"[+] Interactive Pareto chart generated: '{output_html}'")

def main():
    json_path = "benchmark_results.json"
    if not os.path.exists(json_path):
        print(f"Error: '{json_path}' not found. Run 'python run_benchmarks.py' first!")
        sys.exit(1)

    with open(json_path) as f:
        results = json.load(f)

    print("\n" + "=" * 85)
    print(" EMPIRICAL PARETO FRONTIER TABLE")
    print("=" * 85)
    print(f"{'Engine':<28} | {'Conc':<5} | {'Throughput':<12} | {'P99 TTFT':<10} | {'Cost / 1M Tok'}")
    print("-" * 85)
    for r in results:
        print(f"{r['engine']:<28} | {r['concurrency']:<5} | {r['throughput_tok_s']:>7.1f} tok/s | {r['ttft_p99_ms']:>6.1f} ms | ${r['cost_per_1m_tokens_usd']:>6.4f}")
    print("=" * 85)

    generate_html_chart(results, "pareto_curve.html")

if __name__ == "__main__":
    main()
