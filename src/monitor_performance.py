"""
Performance monitoring for the AI trading bot.
Handles flexible log formats and provides comprehensive analytics.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

def load_trade_log(log_path='data/trade_log.json', debug=False):
    """Load and parse trade log with error handling."""
    try:
        with open(log_path, 'r') as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, dict):
            entries = data.get('trades', [data])
        else:
            entries = data
        
        print(f"✓ Loaded {len(entries)} log entries")
        
        # Debug: show first entry structure
        if debug and len(entries) > 0:
            print(f"\n🔍 DEBUG - First entry structure:")
            first_entry = entries[0]
            for key, value in first_entry.items():
                val_preview = str(value)[:50] + '...' if len(str(value)) > 50 else str(value)
                print(f"  {key}: {val_preview} (type: {type(value).__name__})")
            print()
        
        return entries
    except FileNotFoundError:
        print(f"⚠ Log file not found: {log_path}")
        return []
    except json.JSONDecodeError:
        print(f"⚠ Invalid JSON in {log_path}")
        return []

def extract_timestamp(entry):
    """Extract timestamp from log entry (handles multiple formats)."""
    # Try different timestamp key names
    for key in ['timestamp', 'run_time', 'date', 'datetime', 'time']:
        if key in entry:
            ts = entry[key]
            
            # If already datetime, return it
            if isinstance(ts, datetime):
                return ts
            
            # If string, try to parse it
            if isinstance(ts, str):
                # Try different formats
                formats_to_try = [
                    '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%d %H:%M',
                    '%Y-%m-%d',
                    '%Y-%m-%dT%H:%M:%S',
                    '%Y-%m-%dT%H:%M:%S.%f',
                    '%m/%d/%Y %H:%M:%S',
                    '%d/%m/%Y %H:%M:%S',
                ]
                
                for fmt in formats_to_try:
                    try:
                        # Truncate string if needed for format
                        if '%f' in fmt:
                            ts_to_parse = ts[:26]  # Include microseconds
                        elif '%H:%M:%S' in fmt:
                            ts_to_parse = ts[:19]  # Include seconds
                        elif '%H:%M' in fmt:
                            ts_to_parse = ts[:16]  # Include minutes
                        else:
                            ts_to_parse = ts[:10]  # Date only
                        
                        return datetime.strptime(ts_to_parse, fmt)
                    except (ValueError, IndexError):
                        continue
            
            # If numeric timestamp (unix epoch)
            if isinstance(ts, (int, float)):
                return datetime.fromtimestamp(ts)
    
    return None

def parse_log_entries(entries):
    """Parse log entries into structured data."""
    parsed = []
    skipped = 0
    
    for i, entry in enumerate(entries):
        timestamp = extract_timestamp(entry)
        
        if not timestamp:
            skipped += 1
            if skipped <= 3:  # Only print first few warnings
                print(f"  ⚠ Entry {i+1}: No valid timestamp found. Keys: {list(entry.keys())}")
            continue
        
        if not isinstance(timestamp, datetime):
            skipped += 1
            if skipped <= 3:
                print(f"  ⚠ Entry {i+1}: Timestamp is not datetime object (type: {type(timestamp)})")
            continue
            
        record = {
            'date': timestamp.strftime('%Y-%m-%d'),
            'timestamp': timestamp,
            'cash': entry.get('cash_available', entry.get('cash', 0)),
            'portfolio_value': entry.get('portfolio_value', entry.get('equity', 0)),
            'positions': entry.get('current_positions', []),
            'signals': entry.get('signals', {}),
        }
        
        # Count trades executed
        trades_executed = 0
        for key, value in entry.items():
            if isinstance(value, str) and ('BUY' in value or 'SELL' in value):
                if 'shares' in value:  # Actual execution, not just signal
                    trades_executed += 1
        
        record['trades_executed'] = trades_executed
        parsed.append(record)
    
    if skipped > 0:
        print(f"  ℹ Skipped {skipped} entries with invalid timestamps\n")
    
    return parsed

def calculate_metrics(df):
    """Calculate trading performance metrics."""
    if len(df) == 0:
        return {}
    
    metrics = {}
    
    # Portfolio metrics
    initial_value = df.iloc[0]['portfolio_value']
    current_value = df.iloc[-1]['portfolio_value']
    
    metrics['initial_value'] = initial_value
    metrics['current_value'] = current_value
    metrics['total_return'] = ((current_value - initial_value) / initial_value) * 100
    
    # Daily returns
    df['daily_return'] = df['portfolio_value'].pct_change()
    
    # Risk metrics
    metrics['volatility'] = df['daily_return'].std() * np.sqrt(252)  # Annualized
    metrics['sharpe_ratio'] = (df['daily_return'].mean() / df['daily_return'].std()) * np.sqrt(252) if df['daily_return'].std() > 0 else 0
    
    # Drawdown
    df['cummax'] = df['portfolio_value'].cummax()
    df['drawdown'] = (df['portfolio_value'] - df['cummax']) / df['cummax']
    metrics['max_drawdown'] = df['drawdown'].min() * 100
    
    # Trading activity
    metrics['total_days'] = len(df)
    metrics['total_trades'] = df['trades_executed'].sum()
    metrics['avg_trades_per_day'] = metrics['total_trades'] / metrics['total_days']
    
    # Win rate (days portfolio increased)
    winning_days = (df['daily_return'] > 0).sum()
    metrics['win_rate'] = (winning_days / (len(df) - 1)) * 100 if len(df) > 1 else 0
    
    return metrics

def plot_performance(df, save_path='data/performance_report.png'):
    """Create performance visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('AI Trading Bot Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Portfolio Value Over Time
    ax1 = axes[0, 0]
    ax1.plot(df['date'], df['portfolio_value'], 'b-', linewidth=2, label='Portfolio Value')
    ax1.axhline(y=df.iloc[0]['portfolio_value'], color='gray', linestyle='--', alpha=0.5, label='Initial Value')
    ax1.set_title('Portfolio Value')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Daily Returns Distribution
    ax2 = axes[0, 1]
    returns = df['daily_return'].dropna() * 100
    ax2.hist(returns, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_title('Daily Returns Distribution')
    ax2.set_xlabel('Return (%)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    # 3. Drawdown
    ax3 = axes[1, 0]
    drawdown = (df['portfolio_value'] / df['portfolio_value'].cummax() - 1) * 100
    ax3.fill_between(df['date'], drawdown, 0, color='red', alpha=0.3)
    ax3.plot(df['date'], drawdown, 'r-', linewidth=1.5)
    ax3.set_title('Drawdown Over Time')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Drawdown (%)')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Trading Activity
    ax4 = axes[1, 1]
    ax4.bar(df['date'], df['trades_executed'], color='green', alpha=0.6)
    ax4.set_title('Daily Trading Activity')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Number of Trades')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Chart saved to {save_path}")
    plt.close()

def print_report(metrics):
    """Print formatted performance report."""
    print("\n" + "="*60)
    print("PERFORMANCE REPORT".center(60))
    print("="*60)
    
    print(f"\n📊 PORTFOLIO METRICS")
    print(f"  Initial Value:     ${metrics['initial_value']:,.2f}")
    print(f"  Current Value:     ${metrics['current_value']:,.2f}")
    print(f"  Total Return:      {metrics['total_return']:+.2f}%")
    
    print(f"\n📈 RISK METRICS")
    print(f"  Volatility (ann.): {metrics['volatility']:.2f}%")
    print(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:      {metrics['max_drawdown']:.2f}%")
    
    print(f"\n🎯 TRADING ACTIVITY")
    print(f"  Total Days:        {metrics['total_days']}")
    print(f"  Total Trades:      {metrics['total_trades']}")
    print(f"  Avg Trades/Day:    {metrics['avg_trades_per_day']:.1f}")
    print(f"  Win Rate:          {metrics['win_rate']:.1f}%")
    
    print("\n" + "="*60)
    
    # Performance assessment
    if metrics['total_days'] < 5:
        print("⏳ Not enough data yet - keep running the bot!")
    elif metrics['total_return'] > 0 and metrics['max_drawdown'] > -5:
        print("✅ System is performing well - controlled risk, positive returns")
    elif metrics['max_drawdown'] < -10:
        print("⚠️  High drawdown detected - review risk management")
    else:
        print("📊 Continue monitoring - performance within normal ranges")
    
    print("="*60 + "\n")

def monitor(debug=False):
    """Main monitoring function."""
    print("\n🔍 AI Trading Bot Performance Monitor\n")
    
    # Load log
    entries = load_trade_log(debug=debug)
    if not entries:
        print("No data to analyze. Run the bot first to generate trades.")
        return
    
    # Parse entries
    parsed = parse_log_entries(entries)
    if not parsed:
        print("\n⚠ Could not parse any log entries.")
        print("💡 Try running with debug mode to see log structure:")
        print("   Run: python monitor_performance.py --debug\n")
        return
    
    # Create DataFrame
    df = pd.DataFrame(parsed)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    print(f"📅 Analysis period: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    
    # Calculate metrics
    metrics = calculate_metrics(df)
    
    # Generate report
    print_report(metrics)
    
    # Create visualization
    if len(df) >= 2:
        plot_performance(df)
    else:
        print("⏳ Need at least 2 days of data to generate charts")

if __name__ == '__main__':
    import sys
    debug = '--debug' in sys.argv or '-d' in sys.argv
    monitor(debug=debug)