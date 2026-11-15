def calc_ema(price, lookback=20):
    ema = price.ewm(span=lookback, adjust=False).mean()
    return ema

def calc_rsi(price, lookback=14):
    delta = price.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = _calc_sma(gain, lookback)
    avg_loss = _calc_sma(loss, lookback)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)
    return rsi

def calc_macd(price, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(price, fast)
    ema_slow = calc_ema(price, slow)

    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    macd_histogram = macd_line - signal_line
    return macd_histogram

def calc_bb(price, lookback=20):
    rolling_std = price.rolling(window=lookback, min_periods=lookback).std()
    sma = _calc_sma(price, lookback)

    top_band = sma + (2 * rolling_std)
    bottom_band = sma - (2 * rolling_std)
    bb_percentage = (price - bottom_band) / (top_band - bottom_band)
    return bb_percentage

def calc_stochastic_indicator(high, low, close, lookback, d_lookback=3):
    lowest_low = low.rolling(window=lookback, min_periods=lookback).min()
    highest_high = high.rolling(window=lookback, min_periods=lookback).max()

    percent_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    return percent_k

def _calc_sma(price, lookback=20):
    sma = price.rolling(window=lookback, min_periods=lookback).mean()
    return sma

def calc_macd_graph(price, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(price, fast)
    ema_slow = calc_ema(price, slow)

    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram

def calc_bb_graph(price, lookback=20):
    rolling_std = price.rolling(window=lookback, min_periods=lookback).std()
    sma = _calc_sma(price, lookback)

    top_band = sma + (2 * rolling_std)
    bottom_band = sma - (2 * rolling_std)
    bb_percentage = (price - bottom_band) / (top_band - bottom_band)
    return top_band, bottom_band, bb_percentage


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "etan74"

def study_group():
    return author()