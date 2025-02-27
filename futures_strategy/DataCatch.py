import ccxt
import pandas as pd
import datetime
import pytz
import os
from datetime import timedelta
from time import sleep
import asyncio


def calculate_start_time(current_time_ms, interval_minutes, num_intervals):
    # Convert current time from milliseconds to a datetime object
    current_time = datetime.datetime.utcfromtimestamp(current_time_ms / 1000.0)

    # Calculate the total duration to subtract
    total_duration = timedelta(minutes=interval_minutes * num_intervals)

    # Calculate the start time by subtracting the duration from the current time
    start_time = current_time - total_duration

    # Convert the start time back to milliseconds since the epoch
    start_time_ms = int(start_time.timestamp() * 1000)

    return start_time_ms


async def dataCatch(timeframe, num_intervals, futures_symbol):
    # Initialize the Binance exchange with ccxt
    exchange = ccxt.binance({
        'apiKey': '0irxPyL1G7cy3cJoIOyNegZlWemRFRetMgeZ046Iud6oCQmvoLFVkYeDry3O82t5',
        'secret': 'URQsMETBGblyNKWNnlqsRtVQHyBgJ80iqATf9DPPlhB9K8yGBpj3xpOS7TOZBByU',
        'enableRateLimit': True  # Recommended by ccxt when making frequent requests
    })

    # Define the trading pair and contract type
    # futures_symbol = "BTCUSDT"
    contract_type = 'PERPETUAL'

    num_intervals = num_intervals + 299 - 780
    interval_minutes = int(timeframe[:-1])

    # Get current UTC time and set end time
    now = datetime.datetime.utcnow()
    now = now.replace(second=0, microsecond=0)
    until = exchange.parse8601(now.isoformat() + 'Z')

    data = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    since = calculate_start_time(until, interval_minutes, num_intervals)
    while since < until:
        kline_data = None
        try:
            kline_data = exchange.fapiPublicGetContinuousKlines({
                "pair": futures_symbol,
                "contractType": contract_type,
                "interval": timeframe,
                "startTime": since
            })
        except Exception as e:
            print(f"problem fetch kline: {str(e)}")
            await asyncio.sleep(0.1)  # Wait for a second before retrying

        if kline_data:

            # Prepare kline data for DataFrame
            formatted_data = [
                [entry[0], entry[1], entry[2], entry[3], entry[4], entry[5]] for
                entry in kline_data]

            # Create a DataFrame from the formatted data
            formatted_df = pd.DataFrame(formatted_data,
                                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'], dtype=float)

            # Convert timestamps from milliseconds to more readable format
            formatted_df['timestamp'] = pd.to_datetime(formatted_df['timestamp'], unit='ms')
            formatted_df['timestamp'] = formatted_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

            # concat formatted_df into data
            data = pd.concat([data, formatted_df], ignore_index=True)
            # Update since for the next call
            since = int(kline_data[-1][0]) + 1  # Use the last close time plus one millisecond as the new start time
            a = 1
        else:
            continue  # If no data is returned, exit the loop

    return data
