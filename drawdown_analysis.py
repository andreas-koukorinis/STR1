"""
Advanced drawdown and run-up analysis for credit RV framework.

This module provides sophisticated analysis of drawdown episodes, run-up periods,
and trade counting with detailed statistics for the credit RV framework.
"""

from __future__ import annotations
from typing import Sequence, Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd


def _to_np(a) -> np.ndarray:
    """Convert input to numpy array with float dtype and reshape to 1D."""
    return np.asarray(a, dtype=float).reshape(-1)


def trade_count(pnl: Sequence[float]) -> Dict[str, float]:
    """
    Count how many trades, with breakdown.
    
    Args:
        pnl: per-trade PnL (wins > 0, losses < 0, breakeven == 0)
        
    Returns:
        Dictionary containing:
            - total: total number of trades
            - wins: number of winning trades
            - losses: number of losing trades
            - breakeven: number of breakeven trades
            - win_rate: win rate as fraction
    """
    p = _to_np(pnl)
    p = p[np.isfinite(p)]
    total = p.size
    if total == 0:
        return {"total": 0, "wins": 0, "losses": 0, "breakeven": 0, "win_rate": np.nan}
    wins = int(np.sum(p > 0))
    losses = int(np.sum(p < 0))
    breakeven = int(np.sum(p == 0))
    win_rate = wins / total
    return {"total": total, "wins": wins, "losses": losses, "breakeven": breakeven, "win_rate": float(win_rate)}


def equity_from_returns(returns: Sequence[float], initial_capital: float = 1.0) -> np.ndarray:
    """Builds equity curve from period returns (NaNs treated as 0 return).
    
    Args:
        returns: sequence of period returns
        initial_capital: starting capital (default: 1.0)
        
    Returns:
        Cumulative equity curve
    """
    r = _to_np(returns)
    r = np.where(np.isfinite(r), r, 0.0)
    return initial_capital * np.cumprod(1.0 + r)


def _segments_from_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Return list of (start, end) indices for contiguous True segments in a boolean mask.
    
    Args:
        mask: boolean array
        
    Returns:
        List of (start, end) tuples for contiguous True segments
    """
    if mask.size == 0:
        return []
    # Find changes
    diff = np.diff(mask.astype(int))
    starts = list(np.where(diff == 1)[0] + 1)
    ends = list(np.where(diff == -1)[0])
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [mask.size - 1]
    return list(zip(starts, ends))


def drawdown_durations(
    returns: Optional[Sequence[float]] = None,
    equity: Optional[Sequence[float]] = None,
    tol: float = 1e-10,
) -> Dict[str, Any]:
    """
    Compute drawdown episodes and durations from an equity curve.
    
    Provide either returns or equity (currency units). If returns provided, 
    equity is built from 1.0 start.
    
    Definitions:
      - Underwater u_t = equity / rolling_peak - 1 <= 0
      - A drawdown episode is a contiguous run where u_t < -tol (strictly underwater).
      - Peak index = last index before the underwater run where u_t >= -tol (typically u_t ~ 0).
      - Trough index = index of minimum u_t within the episode.
      - Recovery index = first index after the episode where u_t >= -tol (if any).
    
    Durations (in periods):
      - time_to_trough = trough_idx - peak_idx
      - underwater_length = episode_end - episode_start + 1  (bars underwater)
      - peak_to_recovery_duration = recovery_idx - peak_idx  (includes the peak bar; None if unrecovered)
    
    Args:
        returns: optional sequence of period returns
        equity: optional equity curve
        tol: tolerance for underwater definition
        
    Returns:
        Dictionary containing:
            - episodes: list of dicts per drawdown with indices, depths, and durations
            - max_drawdown_duration: maximum peak-to-recovery duration among recovered episodes
            - current_drawdown_length: if still underwater at end, time since last peak; else 0
            - average_drawdown_duration: average over recovered episodes (NaN if none)
    """
    if equity is None and returns is None:
        raise ValueError("Provide either returns or equity.")
    if equity is None:
        e = equity_from_returns(returns, initial_capital=1.0)
    else:
        e = _to_np(equity)
    if e.size == 0 or not np.all(np.isfinite(e)):
        return {
            "episodes": [],
            "max_drawdown_duration": np.nan,
            "current_drawdown_length": np.nan,
            "average_drawdown_duration": np.nan,
        }

    peaks = np.maximum.accumulate(e)
    u = (e / peaks) - 1.0  # underwater (<= 0 at peaks, negative under water)

    underwater_mask = u < -tol
    segments = _segments_from_mask(underwater_mask)

    episodes = []
    durations = []
    for (start, end) in segments:
        peak_idx = max(0, start - 1)  # last non-underwater bar (or 0 if starts at index 0)
        # Trough is point of most negative underwater within [start, end]
        local_idx = int(np.argmin(u[start : end + 1]))
        trough_idx = start + local_idx

        # Recovery: first index after 'end' where not underwater
        recovery_idx = end + 1 if end + 1 < u.size and u[end + 1] >= -tol else None
        # depth as positive fraction
        depth = -float(np.min(u[start : end + 1]))

        time_to_trough = trough_idx - peak_idx
        underwater_length = (end - start + 1)
        peak_to_recovery = (recovery_idx - peak_idx) if recovery_idx is not None else None

        if peak_to_recovery is not None:
            durations.append(peak_to_recovery)

        episodes.append(
            {
                "peak_idx": int(peak_idx),
                "start_underwater_idx": int(start),
                "trough_idx": int(trough_idx),
                "end_underwater_idx": int(end),
                "recovery_idx": (int(recovery_idx) if recovery_idx is not None else None),
                "depth_fraction": depth,            # e.g., 0.23 for -23%
                "depth_percent": depth * 100.0,     # e.g., 23.0
                "time_to_trough": int(time_to_trough),
                "underwater_length": int(underwater_length),
                "peak_to_recovery_duration": (int(peak_to_recovery) if peak_to_recovery is not None else None),
            }
        )

    # Max duration among recovered episodes
    max_drawdown_duration = int(max(durations)) if len(durations) > 0 else np.nan

    # Current drawdown length: if last segment is still in progress (i.e., underwater at end)
    if underwater_mask[-1]:
        last_start, last_end = segments[-1]
        peak_idx = max(0, last_start - 1)
        current_drawdown_length = (u.size - 1) - peak_idx + 0  # inclusive of last bar and peak bar
        current_drawdown_length = int(current_drawdown_length)
    else:
        current_drawdown_length = 0

    average_drawdown_duration = (float(np.mean(durations)) if len(durations) > 0 else np.nan)

    return {
        "episodes": episodes,
        "max_drawdown_duration": max_drawdown_duration,
        "current_drawdown_length": current_drawdown_length,
        "average_drawdown_duration": average_drawdown_duration,
    }


def runup_durations(
    returns: Optional[Sequence[float]] = None,
    equity: Optional[Sequence[float]] = None,
    tol: float = 1e-10,
) -> Dict[str, Any]:
    """
    Compute run-up episodes as contiguous periods where equity is at new highs.
    
    A bar is considered a 'new high' if equity >= rolling_peak * (1 - tol).
    
    Args:
        returns: optional sequence of period returns
        equity: optional equity curve
        tol: tolerance for new high definition
        
    Returns:
        Dictionary containing:
            - episodes: list of dicts with start_idx, end_idx, length
            - max_runup_duration, current_runup_duration, average_runup_duration
    """
    if equity is None and returns is None:
        raise ValueError("Provide either returns or equity.")
    if equity is None:
        e = equity_from_returns(returns, initial_capital=1.0)
    else:
        e = _to_np(equity)
    if e.size == 0 or not np.all(np.isfinite(e)):
        return {
            "episodes": [],
            "max_runup_duration": np.nan,
            "current_runup_duration": np.nan,
            "average_runup_duration": np.nan,
        }

    peaks = np.maximum.accumulate(e)
    # New high if within tolerance of the rolling peak (accounts for floating-point noise)
    new_high_mask = e >= (peaks * (1.0 - tol))
    segments = _segments_from_mask(new_high_mask)

    episodes = []
    lengths = []
    for (start, end) in segments:
        length = end - start + 1
        lengths.append(length)
        episodes.append({"start_idx": int(start), "end_idx": int(end), "length": int(length)})

    max_runup_duration = int(max(lengths)) if len(lengths) > 0 else np.nan
    current_runup_duration = int(lengths[-1]) if len(lengths) > 0 and segments[-1][1] == (e.size - 1) else 0
    average_runup_duration = float(np.mean(lengths)) if len(lengths) > 0 else np.nan

    return {
        "episodes": episodes,
        "max_runup_duration": max_runup_duration,
        "current_runup_duration": current_runup_duration,
        "average_runup_duration": average_runup_duration,
    }


def calculate_advanced_drawdown_metrics(pnl: pd.Series, equity: pd.Series) -> Dict[str, Any]:
    """
    Calculate comprehensive drawdown and run-up metrics for credit RV analysis.
    
    Args:
        pnl: Daily PnL series
        equity: Cumulative equity series
        
    Returns:
        Dictionary with comprehensive drawdown and run-up analysis
    """
    # Convert to numpy arrays
    pnl_array = pnl.values
    equity_array = equity.values
    
    # Basic drawdown analysis
    dd_analysis = drawdown_durations(equity=equity_array)
    ru_analysis = runup_durations(equity=equity_array)
    
    # Trade analysis (using PnL as trades)
    trade_analysis = trade_count(pnl_array)
    
    # Additional statistics
    if len(dd_analysis['episodes']) > 0:
        dd_depths = [ep['depth_percent'] for ep in dd_analysis['episodes']]
        dd_durations = [ep['peak_to_recovery_duration'] for ep in dd_analysis['episodes'] if ep['peak_to_recovery_duration'] is not None]
        
        dd_stats = {
            'num_episodes': len(dd_analysis['episodes']),
            'avg_depth': np.mean(dd_depths),
            'max_depth': np.max(dd_depths),
            'avg_duration': np.mean(dd_durations) if dd_durations else np.nan,
            'total_underwater_time': sum(ep['underwater_length'] for ep in dd_analysis['episodes']),
            'underwater_ratio': sum(ep['underwater_length'] for ep in dd_analysis['episodes']) / len(equity_array)
        }
    else:
        dd_stats = {
            'num_episodes': 0,
            'avg_depth': np.nan,
            'max_depth': np.nan,
            'avg_duration': np.nan,
            'total_underwater_time': 0,
            'underwater_ratio': 0.0
        }
    
    if len(ru_analysis['episodes']) > 0:
        ru_lengths = [ep['length'] for ep in ru_analysis['episodes']]
        ru_stats = {
            'num_episodes': len(ru_analysis['episodes']),
            'avg_length': np.mean(ru_lengths),
            'max_length': np.max(ru_lengths),
            'total_runup_time': sum(ep['length'] for ep in ru_analysis['episodes']),
            'runup_ratio': sum(ep['length'] for ep in ru_analysis['episodes']) / len(equity_array)
        }
    else:
        ru_stats = {
            'num_episodes': 0,
            'avg_length': np.nan,
            'max_length': np.nan,
            'total_runup_time': 0,
            'runup_ratio': 0.0
        }
    
    return {
        'drawdown_analysis': dd_analysis,
        'runup_analysis': ru_analysis,
        'trade_analysis': trade_analysis,
        'drawdown_stats': dd_stats,
        'runup_stats': ru_stats
    }


def print_drawdown_summary(metrics: Dict[str, Any]) -> None:
    """
    Print a comprehensive summary of drawdown and run-up analysis.
    
    Args:
        metrics: Dictionary from calculate_advanced_drawdown_metrics
    """
    dd_analysis = metrics['drawdown_analysis']
    ru_analysis = metrics['runup_analysis']
    trade_analysis = metrics['trade_analysis']
    dd_stats = metrics['drawdown_stats']
    ru_stats = metrics['runup_stats']
    
    print("=" * 60)
    print("ADVANCED DRAWDOWN & RUN-UP ANALYSIS")
    print("=" * 60)
    
    print("\n📉 DRAWDOWN ANALYSIS:")
    print(f"  Number of Drawdown Episodes: {dd_stats['num_episodes']}")
    print(f"  Maximum Drawdown Duration:   {dd_analysis['max_drawdown_duration']} periods")
    print(f"  Current Drawdown Length:     {dd_analysis['current_drawdown_length']} periods")
    print(f"  Average Drawdown Duration:   {dd_analysis['average_drawdown_duration']:.1f} periods")
    print(f"  Average Drawdown Depth:      {dd_stats['avg_depth']:.2f}%")
    print(f"  Maximum Drawdown Depth:      {dd_stats['max_depth']:.2f}%")
    print(f"  Total Underwater Time:       {dd_stats['total_underwater_time']} periods")
    print(f"  Underwater Ratio:            {dd_stats['underwater_ratio']:.2%}")
    
    print("\n📈 RUN-UP ANALYSIS:")
    print(f"  Number of Run-up Episodes:   {ru_stats['num_episodes']}")
    print(f"  Maximum Run-up Duration:     {ru_analysis['max_runup_duration']} periods")
    print(f"  Current Run-up Duration:     {ru_analysis['current_runup_duration']} periods")
    print(f"  Average Run-up Duration:     {ru_analysis['average_runup_duration']:.1f} periods")
    print(f"  Average Run-up Length:       {ru_stats['avg_length']:.1f} periods")
    print(f"  Maximum Run-up Length:       {ru_stats['max_length']} periods")
    print(f"  Total Run-up Time:           {ru_stats['total_runup_time']} periods")
    print(f"  Run-up Ratio:                {ru_stats['runup_ratio']:.2%}")
    
    print("\n🎯 TRADE ANALYSIS:")
    print(f"  Total Trades:                {trade_analysis['total']}")
    print(f"  Winning Trades:              {trade_analysis['wins']}")
    print(f"  Losing Trades:               {trade_analysis['losses']}")
    print(f"  Breakeven Trades:            {trade_analysis['breakeven']}")
    print(f"  Win Rate:                    {trade_analysis['win_rate']:.1%}")
    
    # Episode details
    if len(dd_analysis['episodes']) > 0:
        print(f"\n📊 DRAWDOWN EPISODES (Top 5 by Depth):")
        sorted_episodes = sorted(dd_analysis['episodes'], key=lambda x: x['depth_percent'], reverse=True)
        for i, ep in enumerate(sorted_episodes[:5]):
            print(f"  Episode {i+1}: Depth={ep['depth_percent']:.2f}%, Duration={ep['peak_to_recovery_duration']} periods")
    
    if len(ru_analysis['episodes']) > 0:
        print(f"\n📊 RUN-UP EPISODES (Top 5 by Length):")
        sorted_episodes = sorted(ru_analysis['episodes'], key=lambda x: x['length'], reverse=True)
        for i, ep in enumerate(sorted_episodes[:5]):
            print(f"  Episode {i+1}: Length={ep['length']} periods")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Example usage
    rng = np.random.default_rng(42)
    # Simulated daily returns with some trend and noise
    daily_rets = rng.normal(0.0004, 0.01, size=750)

    # Trades PnL sample
    trade_pnl = rng.normal(120, 450, size=200)

    # How many trades
    print("Trade count:", trade_count(trade_pnl))

    # Drawdown durations
    dd = drawdown_durations(returns=daily_rets)
    print("Max drawdown duration (bars):", dd["max_drawdown_duration"])
    print("Current drawdown length (bars):", dd["current_drawdown_length"])
    print("Average drawdown duration (bars):", dd["average_drawdown_duration"])
    print("First drawdown episode example:", dd["episodes"][0] if len(dd["episodes"]) else None)

    # Run-up durations
    ru = runup_durations(returns=daily_rets)
    print("Max run-up duration (bars):", ru["max_runup_duration"])
    print("Current run-up duration (bars):", ru["current_runup_duration"])
    print("Average run-up duration (bars):", ru["average_runup_duration"])
    print("First run-up episode example:", ru["episodes"][0] if len(ru["episodes"]) else None) 