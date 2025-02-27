import ccxt
import datetime
import pandas as pd
import os
import time
import sys
from time import sleep
from copy import deepcopy
import numpy as np
import random
import csv
import argparse
import asyncio
from datetime import timedelta
from telegram.error import TelegramError


async def get_depth_info(exchange, futures_symbol):
    while True:
        try:
            depth = exchange.fapiPublicGetDepth({"symbol": futures_symbol, "limit": 500})
            return depth['bids'], depth['asks']
        except Exception as e:
            print(f"Error fetching depth from server: {e}")
            await asyncio.sleep(0.1)
            continue
    return

def order_book_close_signal(avg_, max_, min_):
    total_sum = sum(avg_) + sum(max_) + sum(min_)
    total_count = len(avg_) + len(max_) + len(min_)
    return total_sum / total_count

def extract_depth_related_ratios(ratio_data):
    # Check if items are tuples, else assume direct float values
    return [value if isinstance(value, float) else value[1] for value in ratio_data]

def analyze_order_depth(bids, asks):
    total_bids = sum(float(bid[1]) for bid in bids)
    total_asks = sum(float(ask[1]) for ask in asks)

    if total_bids > total_asks:
        sentiment = "Bullish"
    else:
        sentiment = "Bearish"

    bid_ask_ratio = total_bids / total_asks if total_asks > 0 else float('inf')

    return {
        "Total Bids": total_bids,
        "Total Asks": total_asks,
        "Bid/Ask Ratio": bid_ask_ratio,
        "Market Sentiment": sentiment
    }

def get_usdt_balance(balance_dict):
    for element in balance_dict:
        if element['asset'] == 'USDT':
            return element['balance']

def extract_OHLC(local_data_15m):
    data_list = [
        local_data_15m['open'].tolist(),
        local_data_15m['high'].tolist(),
        local_data_15m['low'].tolist(),
        local_data_15m['close'].tolist()
    ]
    return data_list

async def sync_server_time(exchange):
    while True:
        try:
            server_time = exchange.fapiPublicGetTime()
            server_time = int(server_time['serverTime'])
            local_time = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
            return server_time - local_time, server_time
        except Exception as e:
            print(f"Error fetching time from server: {e}")
            await asyncio.sleep(0.1)
            continue
    return

async def get_position_status(exchange, futures_symbol, timestamp):
    while True:
        try:
            position_info = exchange.fapiPrivateV2GetPositionRisk(
                {"symbol": futures_symbol, "timestamp": timestamp, "recvWindow": 9000})
            return position_info

        except Exception as e:
            print(f"Failed to get position info: {str(e)}")
            await asyncio.sleep(0.1)
            continue
    return

def convert_binance_server_time(server_time):
    # Binance server time is in milliseconds, so we need to convert it to seconds
    server_time_seconds = server_time / 1000.0
    # Convert to a datetime object
    readable_time = datetime.datetime.fromtimestamp(server_time_seconds)
    # Format the datetime object to a human-readable string
    readable_time_str = readable_time.strftime('%Y-%m-%d %H:%M:%S')
    return readable_time_str

async def get_current_balance(exchange, server_time):
    while True:
        try:
            current_balance = exchange.fapiPrivateV2GetBalance({
                "recvWindow": 9000,
                "timestamp": server_time
            })
            USDT_balance = float(get_usdt_balance(current_balance))
            USDT_balance = round(USDT_balance, 2)
            return USDT_balance
        except Exception as e:
            print(f"Error fetching balance from server: {e}")
            await asyncio.sleep(0.1)
            _, server_time = await sync_server_time(exchange)
            continue
    return


async def cancel_order(exchange, futures_symbol, timestamp, order_ids):
    while True:
        try:  # "recvWindow": 8000
            canceled = exchange.fapiPrivateDeleteOrder({"symbol": futures_symbol, "orderId": order_ids, "timestamp": timestamp,
                                                        "recvWindow": 9000})
            break
        except Exception as e:
            print(f"Failed to cancel ALL order: {str(e)}")
            continue

async def send_text_message(bot, msg, chat_id):
    """
    Send a message "msg" to a telegram user or group specified by "chat_id"
    msg         [str]: Text of the message to be sent. Max 4096 characters after entities parsing.
    chat_id [int/str]: Unique identifier for the target chat or username of the target channel (in the format @channelusername)
    token       [str]: Bot's unique authentication token.
    """

    try:
        await bot.send_message(chat_id=chat_id, text=msg)
        return
    except TelegramError as e:
        print(f"Telegram error: {e}")
    except Exception as e:
        print(f"General error: {e}")


async def post_stop_only_futures_order(exchange, symbol, side, type, quantity, timestamp, stop_price):
    max_retries = 5  # Maximum number of retries before giving up
    retries = 0

    while retries < max_retries:
        if type == "STOP":
            try:
                order_res = exchange.fapiPrivatePostOrder({
                    "symbol": symbol,  # Symbol must be in the format like 'BTCUSDT'
                    "side": side,  # 'BUY' or 'SELL'
                    # "positionSide": 'LONG',  # 'LONG' or 'SHORT', needed for futures
                    "type": type,
                    "quantity": quantity,
                    "price": stop_price,
                    "timestamp": timestamp,
                    "newOrderRespType": "RESULT",
                    "stopPrice": stop_price
                })
                order_id = int(order_res['orderId'])
                return order_id
            except Exception as e:
                print(f"Error fetching main order from server: {e}")
                # sleep(0.1)
                # continue
                return
        elif type == "TAKE_PROFIT_MARKET" or type == "STOP_MARKET":
            try:
                take_profit_order_res = exchange.fapiPrivatePostOrder({
                    "symbol": symbol,
                    "side": side,
                    "type": type,
                    "stopPrice": stop_price,
                    "closePosition": "true",  # Closes the entire position
                    "timestamp": timestamp,
                })
                order_id = int(take_profit_order_res['orderId'])
                return order_id
            except Exception as e:
                print(f"Error fetching fapiPrivatePostOrder from server: {e}")
                # sleep(0.1)
                # continue
                return
        elif type == "MARKET":
            try:
                order_res = exchange.fapiPrivatePostOrder({
                    "symbol": symbol,  # Symbol must be in the format like 'BTCUSDT'
                    "side": side,  # 'BUY' or 'SELL'
                    # "positionSide": 'LONG',  # 'LONG' or 'SHORT', needed for futures
                    "type": type,
                    "quantity": quantity,
                    "timestamp": timestamp,
                    "newOrderRespType": "RESULT",
                })
                order_id = int(order_res['orderId'])
                return order_id
            except Exception as e:
                print(f"Attempt {retries + 1}: Error placing order on server: {e}")
                retries += 1
                await asyncio.sleep(0.1)  # Wait before retrying
                # _, server_time = sync_server_time(exchange)
    print("Max retries exceeded. Order placement failed.")
    return None  # Explicitly return None after exceeding retries

async def post_futures_order(exchange, symbol, side, type, quantity, timestamp, chat_id, bot, stop_price=None):
    max_retries = 5  # Maximum number of retries before giving up
    retries = 0

    while retries < max_retries:
        try:
            if type == "MARKET":
                order_res = exchange.fapiPrivatePostOrder({
                    "symbol": symbol,  # Symbol must be in the format like 'BTCUSDT'
                    "side": side,  # 'BUY' or 'SELL'
                    "type": type,
                    "quantity": quantity,
                    "timestamp": timestamp,
                    "newOrderRespType": "RESULT",
                })
                order_id = int(order_res['orderId'])
                await send_text_message(bot=bot, msg="### Market Open Position Triggered Successfully! ###",
                                        chat_id=chat_id)
                return order_id
            elif type == "TAKE_PROFIT_MARKET" or type == "STOP_MARKET":
                take_profit_order_res = exchange.fapiPrivatePostOrder({
                    "symbol": symbol,
                    "side": side,
                    "type": type,
                    "stopPrice": stop_price,
                    "closePosition": "true",  # Closes the entire position
                    "timestamp": timestamp,
                })
                order_id = int(take_profit_order_res['orderId'])
                return order_id

            elif type == "closePosition":
                type = "MARKET"
                take_profit_order_res = exchange.fapiPrivatePostOrder({
                    "symbol": symbol,
                    "side": side,
                    "type": type,
                    "quantity": quantity,
                    "timestamp": timestamp,
                })
                order_id = int(take_profit_order_res['orderId'])
                await send_text_message(bot=bot, msg="### Closed Position triggered successfully! ###",
                                        chat_id=chat_id)
                return order_id

        except Exception as e:
            print(f"Attempt {retries + 1}: Error placing order on server: {e}")
            retries += 1
            await asyncio.sleep(0.1)  # Wait before retrying
            # _, server_time = sync_server_time(exchange)

    print("Max retries exceeded. Order placement failed.")
    return None  # Explicitly return None after exceeding retries

def is_first_second_of_minute(kline_server_time, minutes):
    # Convert server time from milliseconds to seconds
    server_time_seconds = kline_server_time / 1000.0 / (minutes * 60)
    # Convert to a datetime object
    # dt = datetime.datetime.fromtimestamp(server_time_seconds)
    # Check if the minute is zero
    return int(server_time_seconds)

async def send_image_message(image_path, caption, chat_id, bot):
    """
    Send an image file "image_path" with a caption "caption" to a telegram user or group specified by "chat_id"
    image_path [str]: Path to the image file.
    caption    [str]: Caption text for the image. Max 1024 characters after entities parsing.
    chat_id    [int/str]: Unique identifier for the target chat or username of the target channel (in the format @channelusername).
    bot        [Bot]: Instance of the telegram Bot.
    """
    photo_file = None
    try:
        photo_file = open(image_path, 'rb')
        await bot.send_photo(chat_id=chat_id, photo=photo_file, caption=caption)
        print("Image sent successfully.")
    except TelegramError as e:
        print(f"Telegram error: {e}")
    except Exception as e:
        print(f"General error: {e}")
    finally:
        # It's important to close the file if it's open
        if photo_file:
            photo_file.close()

def filter_max_volume(df):
    # Group by 'timestamp' and aggregate using custom functions
    # Keep the row with the maximum volume for each group
    return df.groupby('timestamp').agg({
        'open': 'first',  # Keeps the 'open' price from the first row in the group
        'high': 'max',   # Maximum of the 'high' column in the group
        'low': 'min',    # Minimum of the 'low' column in the group
        'close': 'last', # Keeps the 'close' price from the last row in the group
        'volume': 'max'  # Maximum volume, this is used to determine which row to keep
    }).reset_index()

async def cancel_orders(exchange, futures_symbol, order_id, timestamp):
    while True:
        try:  # "recvWindow": 8000
            canceled = exchange.fapiPrivateDeleteOrder({"symbol": futures_symbol, "orderId": order_id})
            break
        except Exception as e:
            print(f"Failed to cancel order {order_id}: {str(e)}")
            await asyncio.sleep(0.1)
            continue

async def cancel_ALL_orders(exchange, futures_symbol, timestamp):
    while True:
        try:  # "recvWindow": 8000
            canceled = exchange.fapiPrivateDeleteAllOpenOrders({"symbol": futures_symbol, "timestamp": timestamp,
                                                                "recvWindow": 9000})
            break
        except Exception as e:
            print(f"Failed to cancel ALL order: {str(e)}")
            await asyncio.sleep(0.1)
            continue

async def set_leverage(exchange, symbol, leverage):
    while True:
        try:
            posted_leverage = exchange.fapiPrivatePostLeverage({
                'symbol': symbol,
                'leverage': leverage
            })
            return posted_leverage
        except Exception as e:
            print(f"Error fetching leverage from server: {e}")
            await asyncio.sleep(0.1)
            continue

def modify_kline_output_format(kline_data):
    # Create a dictionary from basis_data for quick lookup


    # Prepare the kline data using the basis data
    formatted_data = [
        pd.Series([
            pd.to_datetime(int(entry[0]), unit='ms'),  # Keep datetime object for the timestamp
            float(entry[1]),  # Convert string to float
            float(entry[2]),
            float(entry[3]),
            float(entry[4]),
            float(entry[5]),
        ],
            index=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        for entry in kline_data
    ]

    return formatted_data[0]   # Assuming you want the first series object, adjust as needed

def ms_to_formatted_date(ms):
    # Convert milliseconds to seconds
    date = datetime.datetime.utcfromtimestamp(ms / 1000.0)
    # Format the date as requested: 'YYYY-MM-DD HH:MM:SS'
    formatted_date = date.strftime('%Y-%m-%d %H:%M:%S')
    return formatted_date

def calculate_keltner_channel_realtime(buffer, ema_period, atr_period, multiplier):
    # Convert the buffer to a DataFrame if it is not already one
    if not isinstance(buffer, pd.DataFrame):
        buffer = np.transpose(buffer).tolist()
        buffer = pd.DataFrame(buffer, columns=['open', 'high', 'low', 'close'])
    # Ensure the buffer has enough data
    if len(buffer) < max(ema_period, atr_period):
        raise ValueError("Buffer size must be at least as large as the longest period used in calculations")

    # Copy the buffer to avoid modifying the original DataFrame
    buffer = buffer.copy()

    # Calculate the EMA of the closing prices
    buffer.loc[:, 'EMA'] = buffer['close'].ewm(span=ema_period, adjust=False).mean()

    # Calculate True Range (TR)
    buffer.loc[:, 'Previous Close'] = buffer['close'].shift(1)
    buffer.loc[:, 'High-Low'] = buffer['high'] - buffer['low']
    buffer.loc[:, 'High-Previous Close'] = abs(buffer['high'] - buffer['Previous Close'])
    buffer.loc[:, 'Low-Previous Close'] = abs(buffer['low'] - buffer['Previous Close'])
    buffer.loc[:, 'TR'] = buffer[['High-Low', 'High-Previous Close', 'Low-Previous Close']].max(axis=1)

    # Calculate the ATR
    buffer.loc[:, 'ATR'] = buffer['TR'].rolling(window=atr_period).mean()

    # Calculate the Keltner Channel
    upper_band = buffer['EMA'].iloc[-1] + (buffer['ATR'].iloc[-1] * multiplier)
    lower_band = buffer['EMA'].iloc[-1] - (buffer['ATR'].iloc[-1] * multiplier)
    center = buffer['EMA'].iloc[-1]

    return upper_band, lower_band, center


def check_dimensions(df, expected_rows):
    # Desired dimensions
    expected_rows = expected_rows
    expected_columns = 6

    # Get the dimensions of the DataFrame
    rows, columns = df.shape

    # Check if dimensions match the expected values
    if (rows != expected_rows) or (columns != expected_columns):
        print(f"Error: Expected dimensions are {expected_rows} rows and {expected_columns} columns, but got {rows} rows and {columns} columns.")
        # sys.exit("Stopping execution due to incorrect dimensions.")

def check_dimensions_initial(df, expected_rows):
    # Desired dimensions
    expected_rows = expected_rows
    expected_columns = 6

    # Get the dimensions of the DataFrame
    rows, columns = df.shape

    # Check if dimensions match the expected values
    if (rows != expected_rows) or (columns != expected_columns):
        print(f"Error: Expected dimensions are {expected_rows} rows and {expected_columns} columns, but got {rows} rows and {columns} columns.")
        sys.exit("Stopping execution due to incorrect dimensions.")

def find_and_buffer(df, time_frame_min):
    real_time_buffer = pd.DataFrame()
    last_index = len(df) - 1

    # Loop through the DataFrame from the last element to the first
    for i in range(last_index, -1, -1):
        current_timestamp = df.iloc[i]['timestamp']
        current_timestamp = pd.Timestamp(current_timestamp)
        current_timestamp = int(current_timestamp.timestamp() / 60)

        if current_timestamp % time_frame_min == 0:
            if i == last_index:
                # If the last element is evenly divisible
                real_time_buffer = df.iloc[[i]]
            else:
                # If another element is found that is evenly divisible
                real_time_buffer = df.iloc[i:last_index+1]
            break

    return real_time_buffer, i

def resample_ohlc(df, start_index, time_frame_min, num_samples):
    resampled_data = []

    for _ in range(num_samples):
        end_index = start_index - time_frame_min + 1
        if end_index < 0:
            break

        sample = df.iloc[end_index:start_index + 1]

        open_price = sample['open'].iloc[0]
        high_price = sample['high'].max()
        low_price = sample['low'].min()
        close_price = sample['close'].iloc[-1]
        volume_sum = sample['volume'].sum()

        date = sample['timestamp'].iloc[0]  # The start timestamp in the interval

        resampled_data.append({
            'timestamp': date,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume_sum
        })

        start_index = end_index - 1

    resampled_data.reverse()  # Reverse the order to have the most recent data at the top
    resampled_df = pd.DataFrame(resampled_data)
    return resampled_df


def buffer_to_ohlc(real_time_buffer):
    open_price = real_time_buffer['open'].iloc[0]
    high_price = real_time_buffer['high'].max()
    low_price = real_time_buffer['low'].min()
    close_price = real_time_buffer['close'].iloc[-1]
    volume_sum = real_time_buffer['volume'].sum()
    date = real_time_buffer['timestamp'].iloc[0]  # The start timestamp in the buffer

    buffer_ohlc = pd.DataFrame([{
        'timestamp': date,
        'open': open_price,
        'high': high_price,
        'low': low_price,
        'close': close_price,
        'volume': volume_sum
    }])

    return buffer_ohlc

async def get_order_status(exchange, futures_symbol, order_id):
    while True:
        try:
            status = exchange.fapiPrivateGetOrder({"symbol": futures_symbol, "orderId": order_id})
            return status
        except Exception as e:
            print(f"get status: {e}")
            await asyncio.sleep(0.1)
            continue

async def get_top_long_short_ratio(exchange, futures_symbol):
    while True:
        try:
            status = exchange.fapidataGetToplongshortaccountratio({"symbol": futures_symbol, "period": '1h', "limit": 8})
            return status
        except Exception as e:
            print(f"get status: {e}")
            await asyncio.sleep(0.1)
            continue

async def get_open_interest(exchange, futures_symbol):
    while True:
        try:
            status = exchange.fetchOpenInterest(futures_symbol)
            return float(status['openInterestAmount'])
        except Exception as e:
            print(f"get status: {e}")
            continue

async def close_all_position_at_order_book(exchange, order_book, new_close, balance, futures_symbol, bot, PS_chat_id):
    print("Order book is not empty, proceeding with closing orders")
    if order_book[0]['order_type'] == "long":
        # Loop order book and calculate the total quantity
        total_quantity = sum(order['order_quantity'] for order in order_book)
        total_quantity = round(total_quantity, 3)
        close_all_long = {
            "order_type": "close_all_long", "close_price": new_close,
            "order_quantity": total_quantity, "order_status": "open", "Balance": balance
        }
        # Save the close all long order to order book text file
        with open('order_book_break.txt', 'a') as f:
            f.write(f"Close All Long Order: {close_all_long}\n")
        _, server_time = await sync_server_time(exchange)
        await post_futures_order(exchange, futures_symbol, "SELL", "closePosition", total_quantity,
                                 server_time, PS_chat_id, bot,
                                 None)


        await send_text_message(bot=bot, msg="### Closed all long orders ###",
                                chat_id=PS_chat_id)
        print("Closed all long orders")


        return True
    elif order_book[0]['order_type'] == "short":
        # Loop order book and calculate the total quantity
        total_quantity = sum(order['order_quantity'] for order in order_book)
        total_quantity = round(total_quantity, 3)
        close_all_short = {
            "order_type": "close_all_short", "close_price": new_close,
            "order_quantity": total_quantity, "order_status": "open", "Balance": balance
        }
        # Save the close all short order to order book text file
        with open('order_book_break.txt', 'a') as f:
            f.write(f"Close All Short Order: {close_all_short}\n")
        _, server_time = await sync_server_time(exchange)
        await post_futures_order(exchange, futures_symbol, "BUY", "closePosition", total_quantity,
                                 server_time, PS_chat_id, bot,
                                 None)


        await send_text_message(bot=bot, msg="### Closed all short orders ###",
                                chat_id=PS_chat_id)
        print("Closed all short orders")


        return True
    
def moving_sl(open_price, close_price, position_type, total_investment):
  

    # Margin used for trading (e.g., 50% of total investment)
    margin_used = total_investment

    # Total BTC controlled by this margin using leverage
    btc_controlled = (margin_used / open_price)

    # Calculate the total transaction cost at entry and exit
    if position_type == "long":
        entry_value = open_price * btc_controlled
        exit_value = close_price * btc_controlled
        profit = (exit_value - entry_value)
    elif position_type == "short":
        entry_value = open_price * btc_controlled
        exit_value = close_price * btc_controlled
        profit = (entry_value - exit_value)

    # Ensure the loss does not exceed the margin used
    if profit < -margin_used:
        profit = -margin_used  # Cap the loss to the margin used

    return profit

# def calculate_start_time(current_time_ms, interval_minutes, num_intervals):
#     # Convert current time from milliseconds to a datetime object
#     current_time = datetime.datetime.utcfromtimestamp(current_time_ms / 1000.0)

#     # Calculate the total duration to subtract
#     total_duration = timedelta(minutes=interval_minutes * num_intervals)

#     # Calculate the start time by subtracting the duration from the current time
#     start_time = current_time - total_duration

#     # Convert the start time back to milliseconds since the epoch
#     start_time_ms = int(start_time.timestamp() * 1000)

#     return start_time_ms

# def dataCatch():
#     # Initialize the Binance exchange with ccxt
#     exchange = ccxt.binance({
#         'apiKey': 'eJcBYsVVtJIOjYjYAhxok2BZR3pCmni37FMRBP9ZFtiKwHKApbx9G3opQ51dBvkO',
#         'secret': '9IG9qvUcN15rG6pG9m5HJvzFanpgUUcRwySpIspTkpeI53tOPVMYyHG3UthB3uDY',
#         'enableRateLimit': True  # Recommended by ccxt when making frequent requests
#     })

#     # Define the trading pair and contract type
#     futures_symbol = "BTCUSDT"
#     contract_type = 'PERPETUAL'
#     timeframe = '1m'

#     # 4321
#     if timeframe == '1m':
#         num_intervals = 2000+300
#         interval_minutes = 1
#     else:
#         num_intervals = 3200
#         interval_minutes = 30
   
#     # Get current UTC time and set end time
#     now = datetime.datetime.utcnow()
#     now = now.replace(second=0, microsecond=0)
#     until = exchange.parse8601(now.isoformat() + 'Z')

        
#     #initialize a empty dataframe
#     data = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
#     since_list = []
#     since = calculate_start_time(until, interval_minutes, num_intervals)
#     while since < until:
#         kline_data = exchange.fapiPublicGetContinuousKlines({
#             "pair": futures_symbol,
#             "contractType": contract_type,
#             "interval": timeframe,
#             "startTime": since
#             # "limit": 1
#         })

#         if kline_data:

#             # Prepare kline data for DataFrame
#             formatted_data = [
#                 [entry[0], entry[1], entry[2], entry[3], entry[4], entry[5]] for
#                 entry in kline_data]

#             # Create a DataFrame from the formatted data
#             formatted_df = pd.DataFrame(formatted_data,
#                                 columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'], dtype=float)

#             # Convert timestamps from milliseconds to more readable format
#             formatted_df['timestamp'] = pd.to_datetime(formatted_df['timestamp'], unit='ms')
#             formatted_df['timestamp'] = formatted_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

#             #concat formatted_df into data
#             data = pd.concat([data, formatted_df],ignore_index=True)


#             # Update since for the next call
#             since = int(kline_data[-1][6])+1  # Use the last close time plus one millisecond as the new start time
#             # since_list.append(since)
            
#         else:
#             break  # If no data is returned, exit the loop

#     return data