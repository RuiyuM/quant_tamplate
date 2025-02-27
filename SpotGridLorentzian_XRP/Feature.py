import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
import ta

def ema(series, n):
    """
    Compute the Exponential Moving Average (EMA) of a series.
    """
    return series.ewm(span=n, adjust=False).mean()

def sma(series, n):
    """
    Compute the Simple Moving Average (SMA) of a series.
    """
    return series.rolling(window=n).mean()

def rma(series, period):
    """
    Compute the Running Moving Average (RMA) of a series.
    """
    alpha = 1.0 / period
    #series is a numpy array, so we need to convert it to a pandas series
    series = pd.Series(series)
    rma = series.ewm(alpha=alpha, adjust=False).mean()
    return rma

def rescale(series, old_min, old_max, new_min, new_max):
    """
    Rescale a series to a new range.
    """
    return ((series - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

def normalize(series, min_val, max_val):
    """
    Normalize a series to a new range.
    """
    # min_val = series.min()
    # max_val = series.max()
    return ((series - min_val) / (max_val - min_val))


def ta_rsi(src, n1, n2):
    """
    Returns the normalized RSI ideal for use in ML algorithms.

    Parameters:
    src (pd.Series): The input series (i.e., the result of the RSI calculation).
    n1 (int): The length of the RSI.
    n2 (int): The smoothing length of the RSI.

    Returns:
    pd.Series: The normalized RSI.
    """


    # Calculate RSI
    rsi = ta.momentum.RSIIndicator(src, window=n1).rsi()
    
    # Apply EMA smoothing
    ema_rsi = rsi.ewm(span=n2, adjust=False).mean()
    # ema_rsi = ta.ema(rsi, n2)
    
    # Rescale from 0-100 to 0-1
    normalized_rsi = ema_rsi / 100
    
    # normalized_rsi = (ema_rsi - ema_rsi.min()) / (ema_rsi.max() - ema_rsi.min())
    
    return normalized_rsi.iloc[-1]

def EMA_SMA_Filter(data, expectedlen):

    src = data['close']
    src = src.astype(float)

    ema_value = src.ewm(span=expectedlen, adjust=False).mean()
    sma_value = src.rolling(window=expectedlen).mean()
    if src.to_numpy()[-1] > ema_value.to_numpy()[-1] and src.to_numpy()[-1] > sma_value.to_numpy()[-1]:
        return True
    elif src.to_numpy()[-1] < ema_value.to_numpy()[-1] and src.to_numpy()[-1] < sma_value.to_numpy()[-1]:
        return False



def volume_oscillator(data, shortlen, longlen, use_Volume_filter):
    """
    Calculates the Volume Oscillator for given volume data.

    Parameters:
    data (pd.DataFrame): DataFrame containing 'volume' data.
    shortlen (int): The length of the short EMA.
    longlen (int): The length of the long EMA.

    Returns:
    pd.Series: The Volume Oscillator as a percentage.
    """
    if not use_Volume_filter:
        return True
    else:
        src = data['volume']
        src = src.astype(float)
        # Calculate short and long EMAs of volume
        short_ema = src.ewm(span=shortlen, adjust=False).mean()
        long_ema = src.ewm(span=longlen, adjust=False).mean()
        # Calculate the Volume Oscillator
        osc = 100 * (short_ema.to_numpy()[-1] - long_ema.to_numpy()[-1]) / long_ema.to_numpy()[-1]
        if osc > 0:
            V_osc = True
        else:
            V_osc = False
        shortlen = shortlen / 3
        longlen = longlen / 3
        short_ema = src.ewm(span=shortlen, adjust=False).mean()
        long_ema = src.ewm(span=longlen, adjust=False).mean()
        # Calculate the Volume Oscillator
        osc_2 = 100 * (short_ema.to_numpy()[-1] - long_ema.to_numpy()[-1]) / long_ema.to_numpy()[-1]
        if osc_2 < 0 and V_osc:
            return False
        else:
            return V_osc

def ta_cci(src,src2,src3, n1, n2):
    """
    Returns the normalized CCI for the last value in the series.

    Parameters:
    src (pd.Series): The input series.
    n1 (int): The length of the CCI.
    n2 (int): The smoothing length of the CCI.

    Returns:
    float: The normalized CCI value for the last point in the series.
    """
    # Calculate CCI
    cci = ta.trend.CCIIndicator(src,src2,src3, window=n1).cci()
    
    # Apply EMA smoothing
    ema_cci = cci.ewm(span=n2, adjust=False).mean()
    
    # # Normalize to 0-1
    # min_val = ema_cci.min()
    # max_val = ema_cci.max()
    # normalized_cci = (ema_cci.iloc[-1] - min_val) / (max_val - min_val)
    
    return ema_cci.to_numpy()

def ta_wt(data, n1, n2):
    """
    Returns the normalized wave trend (WT) for the last value in the series.

    Parameters:
    src (pd.Series): The input series.
    n1 (int): The length for the first EMA.
    n2 (int): The length for the second EMA.

    Returns:
    float: The normalized WT value for the last point in the series.
    """
    src = (data['high'] + data['low'] + data['close']) / 3
    #change the dtype of src to float
    src = src.astype(float)
    # Calculate first EMA
    ema1 = src.ewm(span=n1, adjust=False).mean()
    
    # Calculate second EMA of the absolute difference
    ema2 = abs(src - ema1).ewm(span=n1, adjust=False).mean()
    
    # Calculate CI
    ci = (src - ema1) / (0.015 * ema2)
    
    # Calculate WT1
    wt1 = ci.ewm(span=n2, adjust=False).mean()
    
    # Calculate WT2
    wt2 = wt1.rolling(window=4).mean()
    
    # Normalize
    diff = wt1 - wt2
    # normalized_wt = (diff - diff.min()) / (diff.max() - diff.min())
    #convert diff to numpy array


    return diff.to_numpy()

def true_range(high, low, close):
    """
    Compute the True Range (TR) of a series.
    """
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr

def directional_movement_plus(high, low, prev_high, prev_low):
    """
    Compute the Positive Directional Movement (DM+).
    """
    up_move = high - prev_high
    down_move = prev_low - low
    return np.where((up_move > down_move) & (up_move > 0), up_move, 0)

def directional_movement_minus(high, low, prev_high, prev_low):
    """
    Compute the Negative Directional Movement (DM-).
    """
    up_move = high - prev_high
    down_move = prev_low - low
    return np.where((down_move > up_move) & (down_move > 0), down_move, 0)

def n_adx(high, low, close, length):
    """
    Compute the normalized ADX ideal for use in ML algorithms.
    
    :param high: The input series for the high price.
    :param low: The input series for the low price.
    :param close: The input series for the close price.
    :param length: The length of the ADX.
    :returns: The normalized ADX.
    """
    tr = true_range(high, low, close)
    dm_plus = directional_movement_plus(high, low, high.shift(1), low.shift(1))
    dm_minus = directional_movement_minus(high, low, high.shift(1), low.shift(1))

    tr_smooth = rma(tr, length)
    dm_plus_smooth = rma(pd.Series(dm_plus), length)
    dm_minus_smooth = rma(pd.Series(dm_minus), length)

    di_positive = (dm_plus_smooth.to_numpy() / tr_smooth.to_numpy()) * 100
    di_negative = (dm_minus_smooth.to_numpy() / tr_smooth.to_numpy()) * 100

    dx = (abs(di_positive - di_negative) / (di_positive + di_negative)) * 100
    adx = rma(dx, length)
    
    normalized_adx = rescale(adx, 0, 100, 0, 1)
    return normalized_adx.iloc[-1]


# Function to calculate ATR using the ta library
def calculate_atr(data, length):
    atr_indicator = ta.volatility.AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=length)
    return atr_indicator.average_true_range().iloc[-1]

# Volatility filter function
def filter_volatility(data, min_length=1, max_length=10, use_volatility_filter=True):
    recent_atr = calculate_atr(data, min_length)
    historical_atr = calculate_atr(data, max_length)

   
    
    if not use_volatility_filter:
        return True
    else:
        return recent_atr > historical_atr

def calculate_ema(source, length):
    """
    Calculate the Exponential Moving Average (EMA) of a given source series.

    Parameters:
    source (pd.Series or list or np.ndarray): Input data series
    length (int): The length of the EMA window

    Returns:
    pd.Series: Series containing the EMA values
    """
    source = pd.Series(source)
    alpha = 2 / (length + 1)
    ema_values = np.zeros(len(source))
    ema_values[0] = source[0]  # Initial EMA value

    for i in range(1, len(source)):
        ema_values[i] = alpha * source[i] + (1 - alpha) * ema_values[i-1]

    return pd.Series(ema_values, index=source.index)

# Function to calculate the regime filter
def regime_filter(data, threshold, use_regime_filter):
    # Calculate src based on 'ohlc4' default
    
    if not use_regime_filter:
        return True
    else: 
        src = (data['open'] + data['high'] + data['low'] + data['close']) / 4
        srchigh = data['high'].to_numpy()
        srclow = data['low'].to_numpy()

        value1 = np.zeros(len(src))
        value2 = np.zeros(len(src))
        klmf = np.zeros(len(src))
        src = src.to_numpy()


        for i in range(1, len(src)):
            value1[i] = 0.2 * (src[i] - src[i-1]) + 0.8 * value1[i-1]
            value2[i] = 0.1 * (srchigh[i] - srclow[i]) + 0.8 * value2[i-1]
            omega = abs(value1[i] / value2[i]) if value2[i] != 0 else 0
            alpha = (-omega**2 + np.sqrt(omega**4 + 16 * omega**2)) / 8
            klmf[i] = alpha * src[i] + (1 - alpha) * klmf[i-1]

        abs_curve_slope = abs(klmf[1:] - klmf[:-1])
        abs_curve_slope = np.insert(abs_curve_slope, 0, 0)

        # exponential_average_abs_curve_slope = ta.trend.EMAIndicator(pd.Series(abs_curve_slope), window=200).ema_indicator()
        # exponential_average_abs_curve_slope = ema(pd.Series(abs_curve_slope), 200)
        # exponential_average_abs_curve_slope = ta.trend.ema_indicator(pd.Series(abs_curve_slope), window=200)
        exponential_average_abs_curve_slope = calculate_ema(abs_curve_slope, 200)
        normalized_slope_decline = (abs_curve_slope - exponential_average_abs_curve_slope) / exponential_average_abs_curve_slope
    
        return normalized_slope_decline.iloc[-1] >= threshold
    
def adx_filter(data,  adx_threshold ,use_adx_filter):
    """
    Compute the normalized ADX ideal for use in ML algorithms.
    
    :param high: The input series for the high price.
    :param low: The input series for the low price.
    :param close: The input series for the close price.
    :param length: The length of the ADX.
    :returns: The normalized ADX.
    """
    if not use_adx_filter:
        return True
    else:
        length = 14
        high = data['high']#.to_numpy()
        low = data['low']#.to_numpy()
        close = data['close']#.to_numpy()
        tr = true_range(high, low, close)
        dm_plus = directional_movement_plus(high, low, high.shift(1), low.shift(1))
        dm_minus = directional_movement_minus(high, low, high.shift(1), low.shift(1))

        tr_smooth = rma(tr, length)
        dm_plus_smooth = rma(pd.Series(dm_plus), length)
        dm_minus_smooth = rma(pd.Series(dm_minus), length)

        di_positive = (dm_plus_smooth.to_numpy() / tr_smooth.to_numpy()) * 100
        di_negative = (dm_minus_smooth.to_numpy() / tr_smooth.to_numpy()) * 100

        dx = (abs(di_positive - di_negative) / (di_positive + di_negative)) * 100
        adx = rma(dx, length)   
        return adx.iloc[-1] > adx_threshold    

def rational_quadratic(src, lookback, relative_weight, start_at_bar):
    current_weight = 0.0
    cumulative_weight = 0.0
    size = 1
    src = src.to_numpy()
    
    for i in range(size + start_at_bar):
        y = src[-1-i]
        w = (1 + ((i ** 2) / ((lookback ** 2) * 2 * relative_weight))) ** -relative_weight
        current_weight += y * w
        cumulative_weight += w
    
    yhat = current_weight / cumulative_weight
    return yhat

def gaussian(src, lookback, start_at_bar):
    current_weight = 0.0
    cumulative_weight = 0.0
    size = 1
    src = src.to_numpy()
    for i in range(size + start_at_bar):
        y = src[-1-i]
        w = np.exp(-(i ** 2) / (2 * (lookback ** 2)))
        current_weight += y * w
        cumulative_weight += w
    
    yhat = current_weight / cumulative_weight
    return yhat
    
    
