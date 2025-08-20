import numpy as np
import pandas as pd
from typing import Callable, Dict, Any, List, Tuple, Optional


#===============================================================


from patterns.triangle import detect_triangle
from patterns.wedge import detect_wedge
from patterns.flag import detect_flag
from patterns.head_and_shoulders import detect_head_and_shoulders
from patterns.double_top import detect_double_top
from patterns.triple_top import detect_triple_top
from patterns.cup_with_handle import detect_cup_with_handle
from patterns.quasimodo import detect_quasimodo
from patterns.wolf_wave import detect_wolf_wave

def build_default_registry():
    reg = PatternRegistry()
    reg.register("Triangle", detect_triangle, priority=1)
    reg.register("Wedge", detect_wedge, priority=2)
    reg.register("Flag", detect_flag, priority=3)
    reg.register("HeadAndShoulders", detect_head_and_shoulders, priority=4)
    reg.register("DoubleTop", detect_double_top, priority=5)
    reg.register("TripleTop", detect_triple_top, priority=6)
    reg.register("CupWithHandle", detect_cup_with_handle, priority=7)
    reg.register("Quasimodo", detect_quasimodo, priority=8)
    reg.register("WolfWave", detect_wolf_wave, priority=9)
    return reg



# =============================================================
# Common utilities
# =============================================================

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df['High'], df['Low'], df['Close']
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def ensure_ohlc(df: pd.DataFrame):
    required = {'Open', 'High', 'Low', 'Close'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame must contain columns {required}, missing: {missing}")


# =============================================================
# Standardized signal schema
# =============================================================
# We normalize any detector output to this minimal schema per bar:
# columns: ['entry_long','entry_short','stop_price','tp_price','pattern']
#   entry_*: bool (enter at next bar open by default)
#   stop_price/tp_price: float or NaN (if NaN, backtester will derive defaults)
#   pattern: string label for audit

SignalDF = pd.DataFrame
Adapter = Callable[[pd.DataFrame, Dict[str, Any]], SignalDF]


class PatternRegistry:
    """Registry of pattern detectors with adapters to normalize outputs."""
    def __init__(self):
        self._adapters: Dict[str, Tuple[Adapter, Dict[str, Any]]] = {}
        self.priority: List[str] = []  # earlier wins when multiple signals on same bar

    def register(self, name: str, adapter: Adapter, params: Optional[Dict[str, Any]] = None, *, priority: int = 100):
        if params is None:
            params = {}
        self._adapters[name] = (adapter, params)
        # maintain a priority-ordered list (lower number = higher priority)
        self.priority.append((priority, name))
        self.priority.sort(key=lambda x: x[0])

    def names_by_priority(self) -> List[str]:
        return [n for _, n in self.priority]

    def adapters(self) -> Dict[str, Tuple[Adapter, Dict[str, Any]]]:
        return self._adapters


# =============================================================
# Scanner
# =============================================================

def run_scanner(df: pd.DataFrame, registry: PatternRegistry) -> Dict[str, SignalDF]:
    ensure_ohlc(df)
    out: Dict[str, SignalDF] = {}
    for name, (adapter, params) in registry.adapters().items():
        sig = adapter(df, params)
        # basic validation
        for col in ['entry_long','entry_short','stop_price','tp_price','pattern']:
            if col not in sig.columns:
                raise ValueError(f"Adapter '{name}' must return column '{col}'")
        out[name] = sig
    return out


def merge_signals(signals: Dict[str, SignalDF], priority_order: List[str]) -> SignalDF:
    # Create a combined DataFrame aligned to index
    all_index = None
    for s in signals.values():
        all_index = s.index if all_index is None else all_index.union(s.index)
    combined = pd.DataFrame(index=all_index, columns=['entry_long','entry_short','stop_price','tp_price','pattern'])
    combined[['entry_long','entry_short']] = False
    combined[['stop_price','tp_price','pattern']] = np.nan

    # Fill by priority: first pattern to signal on a bar wins
    for name in priority_order:
        if name not in signals:
            continue
        s = signals[name]
        # long
        mask_l = s['entry_long'].fillna(False) & (~combined['entry_long']) & (~combined['entry_short'])
        combined.loc[mask_l.index[mask_l], 'entry_long'] = True
        combined.loc[mask_l.index[mask_l], 'stop_price'] = s.loc[mask_l, 'stop_price']
        combined.loc[mask_l.index[mask_l], 'tp_price'] = s.loc[mask_l, 'tp_price']
        combined.loc[mask_l.index[mask_l], 'pattern'] = name
        # short
        mask_s = s['entry_short'].fillna(False) & (~combined['entry_long']) & (~combined['entry_short'])
        combined.loc[mask_s.index[mask_s], 'entry_short'] = True
        combined.loc[mask_s.index[mask_s], 'stop_price'] = s.loc[mask_s, 'stop_price']
        combined.loc[mask_s.index[mask_s], 'tp_price'] = s.loc[mask_s, 'tp_price']
        combined.loc[mask_s.index[mask_s], 'pattern'] = name

    return combined


# =============================================================
# Backtest skeleton
# =============================================================

def backtest(
    df: pd.DataFrame,
    entries: SignalDF,
    *,
    fee_bps: float = 3.0,           # one-way fee in basis points
    slippage_bps: float = 2.0,      # one-way slippage in bps, applied on fill price
    risk_per_trade: float = 0.01,   # 1% of equity risked per trade (if stop known)
    initial_equity: float = 100_000.0,
    timeout_bars: int = 80,
    default_rr: float = 2.0,        # RR target if tp missing
    atr_period: int = 14            # default stop if missing: 1*ATR
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Returns (trades, fills, equity_curve)
    - trades: one row per closed trade
    - fills: per-bar fills/positions snapshot (lightweight)
    - equity_curve: Series of equity by bar
    """
    ensure_ohlc(df)
    df = df.copy()
    df['ATR'] = atr(df, period=atr_period)

    # Align indices
    entries = entries.reindex(df.index).fillna({'entry_long': False, 'entry_short': False})

    fee = fee_bps / 10_000.0
    slip = slippage_bps / 10_000.0

    equity = initial_equity
    position = 0  # +1 long, -1 short, 0 flat
    entry_price = np.nan
    stop_level = np.nan
    tp_level = np.nan
    entry_idx = None
    entry_pattern = None
    size = 0.0

    records = []  # closed trades
    fills = []    # bar snapshots

    for i in range(len(df) - 1):  # last bar cannot exit by hi/lo logic cleanly
        ts = df.index[i]
        o, h, l, c = df['Open'].iloc[i+1], df['High'].iloc[i+1], df['Low'].iloc[i+1], df['Close'].iloc[i+1]  # next bar for fills
        cur = df.iloc[i]

        if position == 0:
            # check for new entry signal on current bar, fill at next bar open with slippage/fee
            go_long = bool(entries['entry_long'].iloc[i])
            go_short = bool(entries['entry_short'].iloc[i])
            if go_long or go_short:
                # determine stop/tp defaults
                ent_side = 1 if go_long else -1
                estop = entries['stop_price'].iloc[i]
                etp   = entries['tp_price'].iloc[i]
                atr_v = cur['ATR'] if not np.isnan(cur['ATR']) else 0.0

                if np.isnan(estop):
                    estop = (cur['Close'] - atr_v) if ent_side == 1 else (cur['Close'] + atr_v)
                if np.isnan(etp):
                    rr = default_rr
                    if ent_side == 1:
                        etp = cur['Close'] + rr * (cur['Close'] - estop)
                    else:
                        etp = cur['Close'] - rr * (estop - cur['Close'])

                # position size: risk_per_trade of equity to stop distance from fill price
                fill = o * (1 + slip * ent_side)  # slip increases price for buy, decreases for sell
                risk_per_unit = abs(fill - estop)
                if risk_per_unit <= 0:
                    # fallback tiny size
                    qty = 0.0
                else:
                    qty = max(0.0, (equity * risk_per_trade) / risk_per_unit)

                # apply entry fee
                cost = fill * qty
                equity -= cost * fee

                position = ent_side
                entry_price = fill
                stop_level = estop
                tp_level = etp
                entry_idx = i+1
                entry_pattern = entries['pattern'].iloc[i]
                size = qty

        else:
            # manage open position: check next bar H/L for stop/target hit order (stop priority)
            exit_reason = None
            exit_price = None

            if position == 1:
                # stop
                if l <= stop_level:
                    exit_price = stop_level
                    exit_reason = 'stop'
                # tp
                elif h >= tp_level:
                    exit_price = tp_level
                    exit_reason = 'tp'
            else:  # short
                if h >= stop_level:
                    exit_price = stop_level
                    exit_reason = 'stop'
                elif l <= tp_level:
                    exit_price = tp_level
                    exit_reason = 'tp'

            # timeout
            if exit_reason is None and entry_idx is not None and (i+1 - entry_idx) >= timeout_bars:
                exit_price = o  # timeout at next open
                exit_reason = 'timeout'

            if exit_reason is not None:
                # slippage against us on exit
                exit_price = exit_price * (1 - slip * position)
                pnl = (exit_price - entry_price) * position * size
                # fees
                equity -= abs(exit_price * size) * fee
                equity += pnl

                records.append({
                    'entry_time': df.index[entry_idx],
                    'exit_time': df.index[i+1],
                    'side': 'long' if position == 1 else 'short',
                    'pattern': entry_pattern,
                    'entry': entry_price,
                    'exit': exit_price,
                    'stop': stop_level,
                    'tp': tp_level,
                    'pnl': pnl,
                    'hold_bars': (i+1 - entry_idx),
                    'reason': exit_reason,
                    'size': size,
                    'equity_after': equity,
                })

                # reset
                position = 0
                entry_price = stop_level = tp_level = np.nan
                entry_idx = None
                entry_pattern = None
                size = 0.0

        fills.append({
            'time': df.index[i+1],
            'position': position,
            'equity': equity,
            'entry': entry_price,
            'stop': stop_level,
            'tp': tp_level,
            'pattern': entry_pattern,
            'size': size,
        })

    trades = pd.DataFrame(records).set_index('exit_time') if records else pd.DataFrame(
        columns=['entry_time','side','pattern','entry','exit','stop','tp','pnl','hold_bars','reason','size','equity_after']
    )
    fills_df = pd.DataFrame(fills).set_index('time') if fills else pd.DataFrame(columns=['position','equity','entry','stop','tp','pattern','size'])
    equity_curve = fills_df['equity'] if 'equity' in fills_df else pd.Series(dtype=float)

    return trades, fills_df, equity_curve


# =============================================================
# Adapters for previously provided detectors (examples)
# =============================================================
# IMPORTANT: Replace the `raise NotImplementedError` with real calls to your detectors
# that you pasted earlier in your environment. Each adapter should return the normalized schema.


def _blank(df: pd.DataFrame, name: str) -> SignalDF:
    return pd.DataFrame({
        'entry_long': False,
        'entry_short': False,
        'stop_price': np.nan,
        'tp_price': np.nan,
        'pattern': name
    }, index=df.index)


# --- Example adapter: Triangle ---

def adapter_triangle(df: pd.DataFrame, params: Dict[str, Any]) -> SignalDF:
    # from your previous code: signals = detect_triangle_breakout(df, **params)
    # EXPECTED columns in signals: ['long_entry','short_entry','stop_price','tp_price']
    try:
        signals = detect_triangle_breakout(df, **params)
        out = pd.DataFrame(index=signals.index)
        out['entry_long'] = signals['long_entry'].fillna(False)
        out['entry_short'] = signals['short_entry'].fillna(False)
        out['stop_price'] = signals.get('stop_price', pd.Series(np.nan, index=signals.index))
        out['tp_price'] = signals.get('tp_price', pd.Series(np.nan, index=signals.index))
        out['pattern'] = 'Triangle'
        return out
    except NameError:
        return _blank(df, 'Triangle')


# --- Example adapter: Wedge ---

def adapter_wedge(df: pd.DataFrame, params: Dict[str, Any]) -> SignalDF:
    try:
        signals = detect_wedge_breakout(df, **params)
        out = pd.DataFrame(index=signals.index)
        out['entry_long'] = signals['long_entry'].fillna(False)
        out['entry_short'] = signals['short_entry'].fillna(False)
        out['stop_price'] = signals.get('stop_price', pd.Series(np.nan, index=signals.index))
        out['tp_price'] = signals.get('tp_price', pd.Series(np.nan, index=signals.index))
        out['pattern'] = signals.get('wedge_type', pd.Series('Wedge', index=signals.index))
        return out
    except NameError:
        return _blank(df, 'Wedge')


# --- Example adapter: Flag ---

def adapter_flag(df: pd.DataFrame, params: Dict[str, Any]) -> SignalDF:
    try:
        signals = detect_flag_breakout(df, **params)
        out = pd.DataFrame(index=signals.index)
        out['entry_long'] = signals['long_entry'].fillna(False)
        out['entry_short'] = signals['short_entry'].fillna(False)
        out['stop_price'] = signals.get('stop_price', pd.Series(np.nan, index=signals.index))
        out['tp_price'] = signals.get('tp_price', pd.Series(np.nan, index=signals.index))
        out['pattern'] = signals.get('flag_type', pd.Series('Flag', index=signals.index))
        return out
    except NameError:
        return _blank(df, 'Flag')


# --- Example adapter: Head & Shoulders ---

def adapter_hns(df: pd.DataFrame, params: Dict[str, Any]) -> SignalDF:
    try:
        signals = detect_head_and_shoulders(df, **params)
        out = pd.DataFrame(index=signals.index)
        out['entry_long'] = signals['long_entry'].fillna(False)
        out['entry_short'] = signals['short_entry'].fillna(False)
        out['stop_price'] = signals.get('stop_price', pd.Series(np.nan, index=signals.index))
        out['tp_price'] = signals.get('tp_price', pd.Series(np.nan, index=signals.index))
        out['pattern'] = signals.get('pattern', pd.Series('HnS', index=signals.index))
        return out
    except NameError:
        return _blank(df, 'HnS')


# --- Example adapter: Double Top/Bottom ---

def adapter_double(df: pd.DataFrame, params: Dict[str, Any]) -> SignalDF:
    try:
        signals = detect_double_top_bottom(df, **params)
        out = pd.DataFrame(index=signals.index)
        out['entry_long'] = signals['entry_long'].fillna(False)
        out['entry_short'] = signals['entry_short'].fillna(False)
        out['stop_price'] = signals.get('stop_price', pd.Series(np.nan, index=signals.index))
        out['tp_price'] = signals.get('tp_price', pd.Series(np.nan, index=signals.index))
        out['pattern'] = signals.get('pattern', pd.Series('Double', index=signals.index))
        return out
    except NameError:
        return _blank(df, 'Double')


# --- Example adapter: Triple Top/Bottom ---

def adapter_triple(df: pd.DataFrame, params: Dict[str, Any]) -> SignalDF:
    try:
        signals = detect_triple_top_bottom(df, **params)
        out = pd.DataFrame(index=signals.index)
        out['entry_long'] = signals['entry_long'].fillna(False)
        out['entry_short'] = signals['entry_short'].fillna(False)
        out['stop_price'] = signals.get('stop_price', pd.Series(np.nan, index=signals.index))
        out['tp_price'] = signals.get('tp_price', pd.Series(np.nan, index=signals.index))
        out['pattern'] = signals.get('pattern', pd.Series('Triple', index=signals.index))
        return out
    except NameError:
        return _blank(df, 'Triple')


# --- Example adapter: Cup with Handle ---

def adapter_cwh(df: pd.DataFrame, params: Dict[str, Any]) -> SignalDF:
    try:
        signals = detect_cup_with_handle(df, **params)
        out = pd.DataFrame(index=signals.index)
        out['entry_long'] = signals['entry_long'].fillna(False)
        out['entry_short'] = signals.get('entry_short', pd.Series(False, index=signals.index))
        out['stop_price'] = signals.get('stop_price', pd.Series(np.nan, index=signals.index))
        out['tp_price'] = signals.get('tp_price', pd.Series(np.nan, index=signals.index))
        out['pattern'] = signals.get('pattern', pd.Series('CupHandle', index=signals.index))
        return out
    except NameError:
        return _blank(df, 'CupHandle')


# --- Example adapter: Quasimodo ---

def adapter_qm(df: pd.DataFrame, params: Dict[str, Any]) -> SignalDF:
    try:
        signals = detect_quasimodo(df, **params)
        out = pd.DataFrame(index=signals.index)
        out['entry_long'] = signals['entry_long'].fillna(False)
        out['entry_short'] = signals['entry_short'].fillna(False)
        out['stop_price'] = signals.get('stop_price', pd.Series(np.nan, index=signals.index))
        out['tp_price'] = signals.get('tp_price', pd.Series(np.nan, index=signals.index))
        out['pattern'] = signals.get('pattern', pd.Series('QM', index=signals.index))
        return out
    except NameError:
        return _blank(df, 'QM')


# --- Example adapter: Wolfe Wave ---

def adapter_wolfe(df: pd.DataFrame, params: Dict[str, Any]) -> SignalDF:
    try:
        signals = detect_wolfe_wave(df, **params)
        out = pd.DataFrame(index=signals.index)
        out['entry_long'] = signals['entry_long'].fillna(False)
        out['entry_short'] = signals['entry_short'].fillna(False)
        out['stop_price'] = signals.get('stop_price', pd.Series(np.nan, index=signals.index))
        out['tp_price'] = signals.get('tp_price', pd.Series(np.nan, index=signals.index))
        out['pattern'] = signals.get('pattern', pd.Series('Wolfe', index=signals.index))
        return out
    except NameError:
        return _blank(df, 'Wolfe')


# =============================================================
# Example: building a registry
# =============================================================

def build_default_registry() -> PatternRegistry:
    reg = PatternRegistry()
    # You can tune params per detector here.
    reg.register('Triangle', adapter_triangle, params=dict(), priority=10)
    reg.register('Wedge', adapter_wedge, params=dict(), priority=20)
    reg.register('Flag', adapter_flag, params=dict(), priority=30)
    reg.register('HnS', adapter_hns, params=dict(), priority=40)
    reg.register('Double', adapter_double, params=dict(), priority=50)
    reg.register('Triple', adapter_triple, params=dict(), priority=60)
    reg.register('CupHandle', adapter_cwh, params=dict(), priority=70)
    reg.register('QM', adapter_qm, params=dict(), priority=80)
    reg.register('Wolfe', adapter_wolfe, params=dict(), priority=90)
    return reg


# =============================================================
# End-to-end convenience
# =============================================================

def scan_and_backtest(
    df: pd.DataFrame,
    registry: Optional[PatternRegistry] = None,
    *,
    fee_bps: float = 3.0,
    slippage_bps: float = 2.0,
    risk_per_trade: float = 0.01,
    initial_equity: float = 100_000.0,
    timeout_bars: int = 80,
    default_rr: float = 2.0,
    atr_period: int = 14
) -> Tuple[Dict[str, SignalDF], SignalDF, pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Returns: (per_pattern_signals, merged_entries, trades, fills, equity)
    """
    if registry is None:
        registry = build_default_registry()
    per = run_scanner(df, registry)
    merged = merge_signals(per, registry.names_by_priority())
    trades, fills, equity = backtest(
        df, merged,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        risk_per_trade=risk_per_trade,
        initial_equity=initial_equity,
        timeout_bars=timeout_bars,
        default_rr=default_rr,
        atr_period=atr_period,
    )
    return per, merged, trades, fills, equity


# =============================================================
# Usage example (uncomment to run after you paste detectors):
# =============================================================
# if __name__ == "__main__":
#     import pandas as pd
#     df = pd.read_csv("ohlc.csv", parse_dates=['Date'], index_col='Date')
#     registry = build_default_registry()
#     per, merged, trades, fills, equity = scan_and_backtest(df, registry)
#     print(merged[merged['entry_long'] | merged['entry_short']].tail(20))
#     print(trades.tail(10))
