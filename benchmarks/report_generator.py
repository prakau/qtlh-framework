#!/usr/bin/env python3
"""
Generate comprehensive benchmark reports for QTL-H Framework.
"""

import json
import os
from datetime import datetime
from pathlib import Path
import yaml

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from jinja2 import Template

class BenchmarkReporter:
    """Generate reports from benchmark results."""

    def __init__(self, results_dir="benchmark_results", config_path="benchmark_config.yaml"):
        self.results_dir = Path(results_dir)
        self.config_path = Path(config_path)
        self.load_config()
        self.results = {}
        self.comparisons = {}
        
    def load_config(self):
        """Load benchmark configuration."""
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
            
    def load_results(self, results_file):
        """Load benchmark results from JSON file."""
        with open(results_file) as f:
            self.results = json.load(f)
            
    def generate_summary_stats(self):
        """Generate summary statistics for all benchmarks."""
        stats = {}
        for module, results in self.results.items():
            stats[module] = {
                'mean': np.mean(results['times']),
                'std': np.std(results['times']),
                'min': np.min(results['times']),
                'max': np.max(results['times']),
                'p95': np.percentile(results['times'], 95),
                'p99': np.percentile(results['times'], 99),
                'samples': len(results['times'])
            }
        return stats
    
    def plot_performance_comparison(self):
        """Create performance comparison plots."""
        fig = go.Figure()
        
        for module, stats in self.generate_summary_stats().items():
            fig.add_trace(go.Box(
                y=self.results[module]['times'],
                name=module,
                boxpoints='outliers'
            ))
            
        fig.update_layout(
            title='Performance Comparison Across Modules',
            yaxis_title='Time (seconds)',
            showlegend=True
        )
        
        return fig
    
    def plot_memory_usage(self):
        """Plot memory usage across modules."""
        memory_data = []
        for module, results in self.results.items():
            if 'memory' in results:
                memory_data.append({
                    'module': module,
                    'peak_memory': results['memory']['peak'],
                    'avg_memory': results['memory']['mean']
                })
                
        df = pd.DataFrame(memory_data)
        fig = px.bar(
            df,
            x='module',
            y=['peak_memory', 'avg_memory'],
            title='Memory Usage by Module',
            barmode='group'
        )
        
        return fig
    
    def generate_regression_analysis(self, baseline_results):
        """Compare current results with baseline for regressions."""
        regressions = {}
        baseline = self.load_baseline_results(baseline_results)
        
        for module in self.results:
            if module in baseline:
                current_mean = np.mean(self.results[module]['times'])
                baseline_mean = np.mean(baseline[module]['times'])
                
                change_pct = ((current_mean - baseline_mean) / baseline_mean) * 100
                regressions[module] = {
                    'change_percentage': change_pct,
                    'significant': abs(change_pct) > self.config['comparison']['max_regression_threshold']
                }
                
        return regressions
    
    def create_html_report(self, output_path):
        """Generate HTML report with all visualizations and analyses."""
        template = Template("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>QTL-H Framework Benchmark Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .section { margin: 20px 0; }
                .warning { color: red; }
                table { border-collapse: collapse; width: 100%; }
                th, td { padding: 8px; text-align: left; border: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>QTL-H Framework Benchmark Report</h1>
            <div class="section">
                <h2>Summary</h2>
                {{ summary_table | safe }}
            </div>
            
            <div class="section">
                <h2>Performance Comparison</h2>
                {{ performance_plot | safe }}
            </div>
            
            <div class="section">
                <h2>Memory Usage</h2>
                {{ memory_plot | safe }}
            </div>
            
            {% if regressions %}
            <div class="section">
                <h2>Performance Regressions</h2>
                {{ regression_table | safe }}
            </div>
            {% endif %}
            
            <div class="section">
                <h2>Test Configuration</h2>
                <pre>{{ config | safe }}</pre>
            </div>
            
            <footer>
                <p>Generated on {{ timestamp }}</p>
            </footer>
        </body>
        </html>
        """)
        
        # Generate plots
        perf_plot = self.plot_performance_comparison()
        mem_plot = self.plot_memory_usage()
        
        # Create summary table
        stats = self.generate_summary_stats()
        summary_df = pd.DataFrame(stats).round(3)
        
        # Generate HTML
        html = template.render(
            summary_table=summary_df.to_html(),
            performance_plot=perf_plot.to_html(full_html=False),
            memory_plot=mem_plot.to_html(full_html=False),
            regression_table=pd.DataFrame(self.comparisons).to_html() if self.comparisons else "",
            config=yaml.dump(self.config, default_flow_style=False),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(html)
            
    def save_results_csv(self, output_path):
        """Save detailed results to CSV."""
        data = []
        for module, results in self.results.items():
            for time in results['times']:
                data.append({
                    'module': module,
                    'time': time,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
    def generate_all_reports(self):
        """Generate all report formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.results_dir / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate HTML report
        self.create_html_report(output_dir / 'report.html')
        
        # Save CSV results
        self.save_results_csv(output_dir / 'results.csv')
        
        # Save JSON results
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
            
        # Generate plots
        perf_plot = self.plot_performance_comparison()
        mem_plot = self.plot_memory_usage()
        
        perf_plot.write_image(output_dir / 'performance.png')
        mem_plot.write_image(output_dir / 'memory.png')
        
        return output_dir

if __name__ == "__main__":
    reporter = BenchmarkReporter()
    reporter.load_results("benchmark_results/latest/results.json")
    output_dir = reporter.generate_all_reports()
    print(f"Reports generated in: {output_dir}")
