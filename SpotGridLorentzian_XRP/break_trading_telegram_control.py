import ccxt
import datetime
import pandas as pd
import os
import time
from time import sleep
from copy import deepcopy

import numpy as np
import random
import csv

from plot_OHLC import plot_ohlc
from telegram.error import TelegramError
import asyncio
import signal
from telegram import Bot, Update
from telegram.ext import Updater
from DataCatch_break import dataCatch
import sys
from receieve_and_send import process_updates



bot_token = '6741244950:AAGl8erp5Kmh67prHEfj7rDHb3jH1THzXgI'
PS_chat_id = '6473165102'
GP_chat_id = '-1002196074683'
bot = Bot(token=bot_token)
##### Telegram command
Tele_com_open_position = "start"
Tele_com_pause_position = "pause"
Tele_com_close_position = "close"
Tele_com_refresh_highest = "refresh_high"
Tele_com_refresh_lowest = "refresh_low"
Tele_com_get_balance = "balance"
Tele_com_make_all_none = "None"


def redefine_args_for_5m_1m(args, path_5m, path_1m):
    dataset_5m = os.path.basename(path_5m)
    dataset_1m = os.path.basename(path_1m)

    # Create a deep copy for 5-minute args
    args_5m = deepcopy(args)
    args_5m.data_path = dataset_5m
    args_5m.model_id = 'BTC_USDT_perpetual_futures_15m'
    args_5m.freq = 'h'
    args_5m.pred_len = 12
    # Create another deep copy for 1-minute args
    args_1m = deepcopy(args)
    args_1m.data_path = dataset_1m
    args_1m.model_id = 'BTC_USDT_perpetual_futures_30m'
    args_1m.freq = 'h'
    args_1m.pred_len = 12

    return args_5m, args_1m


def extract_OHLC(local_data_15m):
    data_list = [
        local_data_15m['open'].tolist(),
        local_data_15m['high'].tolist(),
        local_data_15m['low'].tolist(),
        local_data_15m['close'].tolist()
    ]
    return data_list



def create_setting(args):
    setting_format = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'
    return setting_format.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des,
        0
    )


def find_index_of_high_low_for_kline(kline):
    # Extracting high and low values
    highs = kline[:, 1]
    lows = kline[:, 2]

    # Finding the index of the highest "high" and the lowest "low"
    index_highest_high = np.argmax(highs)
    index_lowest_low = np.argmin(lows)

    return index_highest_high, index_lowest_low, highs[index_highest_high], lows[index_lowest_low]


def find_similar_extremes(data_list, threshold_percentage):
    # Extract the lists for high and low values
    highs = data_list[1]
    lows = data_list[2]

    # Find the highest of highs and the lowest of lows
    highest = max(highs)
    lowest = min(lows)

    # Calculate the threshold values
    high_threshold = highest * threshold_percentage / 100
    low_threshold = lowest * threshold_percentage / 100

    # Count similar values within the threshold for highs
    similar_high_count = sum(1 for high in highs if abs(highest - high) <= high_threshold)

    # Count similar values within the threshold for lows
    similar_low_count = sum(1 for low in lows if abs(lowest - low) <= low_threshold)

    return similar_high_count, similar_low_count


def within_threshold(value1, value2, threshold_percentage):
    # Calculate the threshold based on the first value
    threshold = abs(value1 * threshold_percentage / 100)

    # Check if the absolute difference between the two values is within the threshold
    return abs(value1 - value2) <= threshold


def calculate_tp_sl(open_price, tp_ratio, sl_ratio, position_type):
    if position_type == "long":
        # Calculate take profit (TP) price
        tp_price = open_price * (1 + tp_ratio / 100)

        # Calculate stop loss (SL) price
        sl_price = open_price * (1 - sl_ratio / 100)
    else:
        sl_price = open_price * (1 + sl_ratio / 100)

        # Calculate stop loss (SL) price
        tp_price = open_price * (1 - tp_ratio / 100)

    return tp_price, sl_price


def find_index_of_high_low_for_kline_close_price(kline):
    # Extracting high and low values
    highs = kline[:, 3]
    lows = kline[:, 3]

    # Finding the index of the highest "high" and the lowest "low"
    index_highest_high = np.argmax(highs)
    index_lowest_low = np.argmin(lows)

    return index_highest_high, index_lowest_low, highs[index_highest_high], lows[index_lowest_low]


def load_data(file_path, start_time, end_time):
    """
    Load and filter the data from a CSV file based on the specified time range.

    Parameters:
        file_path (str): The path to the CSV file.
        start_time (str): The starting time for the data in "YYYY-MM-DD HH:MM:SS" format.
        end_time (str): The ending time for the data in "YYYY-MM-DD HH:MM:SS" format.

    Returns:
        pandas.DataFrame: The loaded and filtered data.
    """
    # Load the data with timestamp parsing
    data = pd.read_csv(file_path, parse_dates=['timestamp'])

    # Filter data based on the provided start and end time
    mask = (data['timestamp'] >= start_time) & (data['timestamp'] <= end_time)
    filtered_data = data.loc[mask]

    # Ensure data is sorted by timestamp
    filtered_data.sort_values('timestamp', inplace=True)

    return filtered_data


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

def convert_binance_server_time(server_time):
    # Binance server time is in milliseconds, so we need to convert it to seconds
    server_time_seconds = server_time / 1000.0
    # Convert to a datetime object
    readable_time = datetime.datetime.fromtimestamp(server_time_seconds)
    # Format the datetime object to a human-readable string
    readable_time_str = readable_time.strftime('%Y-%m-%d %H:%M:%S')
    return readable_time_str

def get_usdt_balance(balance_dict):
    for element in balance_dict:
        if element['asset'] == 'USDT':
            return element['balance']

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


async def post_futures_order(exchange, symbol, side, type, quantity, timestamp, stop_price=None):
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
                await send_text_message(msg="### Market Open Position Triggered Successfully! ###",
                                        chat_id=GP_chat_id)
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
                await send_text_message(msg="### Closed Position triggered successfully! ###",
                                        chat_id=GP_chat_id)
                return order_id

        except Exception as e:
            print(f"Attempt {retries + 1}: Error placing order on server: {e}")
            retries += 1
            await asyncio.sleep(0.1)  # Wait before retrying
            # _, server_time = sync_server_time(exchange)

    print("Max retries exceeded. Order placement failed.")
    return None  # Explicitly return None after exceeding retries

def check_position_type(if_currently_have_position, existing_position_type, want_to_open_position_type):
    if existing_position_type is None:
        return True
    elif if_currently_have_position:
        if existing_position_type == want_to_open_position_type:
            return True
        else:
            return False
    else:
        return True


def is_first_minute_of_hour(kline_server_time):
    # Convert server time from milliseconds to seconds
    server_time_seconds = kline_server_time / 1000.0
    # Convert to a datetime object
    dt = datetime.datetime.fromtimestamp(server_time_seconds)
    # Check if the minute is zero
    return dt.minute

def update_take_profit_and_stop_loss(current_tp, current_sl, target_tp, target_sl, position_type):
    if current_tp is None and current_sl is None:
        return target_tp, target_sl

    if position_type == "long":
        # For a long position, take profit should be maximized and stop loss minimized
        new_tp = max(current_tp, target_tp)
        new_sl = max(current_sl, target_sl)
    elif position_type == "short":
        # For a short position, take profit should be minimized and stop loss maximized
        new_tp = min(current_tp, target_tp)
        new_sl = min(current_sl, target_sl)
    else:
        # Return existing values if position type is not recognized
        return current_tp, current_sl

    return new_tp, new_sl


async def send_text_message(msg, chat_id):
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


def get_order_status(exchange, futures_symbol, order_id):
    while True:
        try:
            status = exchange.fapiPrivateGetOrder({"symbol": futures_symbol, "orderId": order_id})
            return status
        except Exception as e:
            print(f"get status: {e}")
            return


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


def update_balance_file(balance, balance_file_path):
    # Check if the file already exists to determine if we need to write the header
    file_exists = os.path.exists(balance_file_path)

    # Open the file in append mode, create if it does not exist
    with open(balance_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # If the file does not exist, write the header
        if not file_exists:
            writer.writerow(['Balance'])

        # Write the updated balance
        writer.writerow([balance])


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

    return formatted_data[0]  # Assuming you want the first series object, adjust as needed


def ms_to_formatted_date(ms):
    # Convert milliseconds to seconds
    date = datetime.datetime.utcfromtimestamp(ms / 1000.0)
    # Format the date as requested: 'YYYY-MM-DD HH:MM:SS'
    formatted_date = date.strftime('%Y-%m-%d %H:%M:%S')
    return formatted_date

def check_dimensions(df):
    # Desired dimensions
    expected_rows = 188
    expected_columns = 6

    # Get the dimensions of the DataFrame
    rows, columns = df.shape

    # Check if dimensions match the expected values
    if (rows != expected_rows) or (columns != expected_columns):
        print(f"Error: Expected dimensions are {expected_rows} rows and {expected_columns} columns, but got {rows} rows and {columns} columns.")
        sys.exit("Stopping execution due to incorrect dimensions.")


# 目前先设置小window 15 * 96 大window 30 * 96
async def backtest_strategy(exchange, leverage, investment_percent, maker_fee_percentage,
                            taker_fee_percentage, futures_symbol, contract_type, threshold_percentage,
                            TP_SL_ratio, loss_per_trade, k1_ratio, k2_ratio, SL_moving_ratio, default_window,
                            command_queue,
                            ):


    timeframe_1h = '1h'

    i = 0

    local_data_1h = dataCatch()
    check_dimensions(local_data_1h)
    # local_data_1h['timestamp'] = pd.to_datetime(local_data_1h['timestamp'])


    setting_leverage = await set_leverage(exchange, futures_symbol, leverage)
    if setting_leverage:
        print(f"setting the leverage to {leverage}")
    ########################################################################################
    ############# average position price

    order_book = []
    

    # telegram controlled variables
    Tele_start_place_order = False
    Tele_close_all = False
    Tele_get_new_highest = False
    Tele_get_new_lowest = False
    Tele_adjust_window_size = default_window


    offset, server_time = await sync_server_time(exchange)
    balance = await get_current_balance(exchange, server_time)

    Kline_first_minute_of_hour = False
    j = 0
    print("start strategy")
    while True:
        offset, server_time = await sync_server_time(exchange)
        current_time = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000) + offset
        # Calculate the next 2-second interval
        next_interval = ((current_time // 2000) + 1) * 2000
        sleep_time = ((next_interval - current_time) / 1000)  # in seconds
        start_time = time.time()
        while (time.time() - start_time) < sleep_time:
            await asyncio.sleep(1)  # Sleep in 1-second intervals to conserve CPU



        offset, server_time = await sync_server_time(exchange)
        # kline_server_time = convert_binance_server_time(server_time)
        if is_first_minute_of_hour(server_time) == 0 and j == 0:
            Kline_first_minute_of_hour = True
            j += 1
        elif is_first_minute_of_hour(server_time) == 0 and j != 0:
            Kline_first_minute_of_hour = False
            j += 1
        elif is_first_minute_of_hour(server_time) != 0:
            j = 0


        if Kline_first_minute_of_hour:
            SL_number = 0

        if not_place_order_for_next_few_min and i < 30:
            i += 1
        elif not_place_order_for_next_few_min and i == 30:
            i = 0
            not_place_order_for_next_few_min = False

        try:
            # command is (str) and sender_id is (int)
            command, sender_id = await asyncio.wait_for(command_queue.get(), timeout=1)
            if command and sender_id:
                if sender_id == int(PS_chat_id):
                    print(f"Received command: {command}")

                    if command.isdigit():
                        Tele_adjust_window_size = int(command)

                        await send_text_message(msg=f"### adjust window size to {Tele_adjust_window_size} ###",
                                                chat_id=GP_chat_id)
                    elif command.lower() == Tele_com_open_position.lower():
                        Tele_start_place_order = True

                        await send_text_message(msg="### start looking for chance to open position ###",
                                                chat_id=GP_chat_id)
                    elif command.lower() == Tele_com_pause_position.lower():
                        Tele_start_place_order = False

                        await send_text_message(msg="### stop looking for chance to open position ###",
                                                chat_id=GP_chat_id)
                    elif command.lower() == Tele_com_close_position.lower():
                        Tele_close_all = True


                    elif command.lower() == Tele_com_refresh_highest.lower():
                        Tele_get_new_highest = True
                        await send_text_message(msg="### refreshed highest price ###",
                                                chat_id=GP_chat_id)
                    elif command.lower() == Tele_com_refresh_lowest.lower():
                        Tele_get_new_lowest = True
                        await send_text_message(msg="### refreshed low price ###",
                                                chat_id=GP_chat_id)
                    elif command.lower() == Tele_com_get_balance.lower():
                        await send_text_message(msg=f"### current balance: {balance} ###",
                                                chat_id=GP_chat_id)
                    elif command.lower() == Tele_com_make_all_none.lower():
                        make_everthing_None = True
                        await send_text_message(msg=f"### make all variables to its initial states ###",
                                                chat_id=GP_chat_id)


            # Process the command and adjust the strategy accordingly
            command_queue.task_done()
        except asyncio.TimeoutError:
            pass
        ############# close all ############################
        if Tele_close_all:
            Tele_close_all = False
            if order_book:
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
                    await post_futures_order(exchange, futures_symbol, "SELL", "closePosition", total_quantity, server_time,
                                       None)

                    order_book = []
                    await send_text_message(msg="### Closed all long orders ###",
                                            chat_id=GP_chat_id)
                    print("Closed all long orders")



                    make_everthing_None = True
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
                    await post_futures_order(exchange, futures_symbol, "BUY", "closePosition", total_quantity, server_time,
                                       None)

                    order_book = []
                    await send_text_message(msg="### Closed all short orders ###",
                                            chat_id=GP_chat_id)
                    print("Closed all short orders")

                    make_everthing_None = True
            else:
                await send_text_message(msg="### Order book is empty, nothing to close ###",
                                        chat_id=GP_chat_id)



        # 做一个容错就是如果这次fetch data 还是上一分钟的就re-fetch
        while True:
            try:

                new_data_1h = exchange.fapiPublicGetContinuousKlines({
                    "pair": futures_symbol,
                    "contractType": contract_type,
                    "interval": timeframe_1h,
                    "limit": 5,
                    # "startTime": since
                })

                row_15 = modify_kline_output_format([new_data_1h[-1]])
                # local_data_15_window
                if Kline_first_minute_of_hour:
                    row_15 = row_15.to_frame().T
                    # local_data_1h = pd.concat([local_data_1h, row_15], ignore_index=True)
                    local_data_1h = local_data_1h.append(row_15, ignore_index=True)
                    # local_data_1h = local_data_1h.sort_values(by='timestamp')
                    local_data_1h = local_data_1h.iloc[1:]


                else:
                    # Replace the last row in local_data_15_window with row_15
                    local_data_1h.iloc[-1] = row_15

                break

            except Exception as e:
                print(f"An error occurred: {str(e)}")
                await asyncio.sleep(0.1)  # Wait for a second before retrying

        max_position_size = min(150, balance * (leverage - 2))
        # loss_per_trade = max(loss_per_trade, balance * 0.001)
        # loss_per_trade = 0.5
        current_local_data_1h = local_data_1h[-Tele_adjust_window_size:].copy()
        extracted_kline = extract_OHLC(current_local_data_1h)

        current_highest = max(extracted_kline[1])

        current_lowest = min(extracted_kline[2])

        high_count, low_count = find_similar_extremes(extracted_kline, threshold_percentage)

        new_open = extracted_kline[0][-1]
        new_close = extracted_kline[3][-1]
        new_high = extracted_kline[1][-1]
        new_low = extracted_kline[2][-1]
        # print(high_count, low_count)

        # if >= two low means the low is about to break:
        if Tele_get_new_highest:
            currently_have_active_post_long_break = False
            Tele_get_new_highest = False
        if Tele_get_new_lowest:
            currently_have_active_post_short_break = False
            Tele_get_new_lowest = False

        if low_count >= 2 and not currently_have_active_post_short_break:
            print(extracted_kline)
            position_type_short_break = "short"
            open_order_short_price_break_flag = current_lowest
            currently_have_active_post_short_break = True
            ##### plot
            plot_time = convert_ms_to_utc_date_str(server_time)
            plot_ohlc(extracted_kline, plot_time, 60, current_lowest - (current_lowest * 0.1 / 100), None)
            image_path = 'ohlc_plot_60.png'
            caption = 'Short: current Kline and potential opening position price.'
            await send_image_message(image_path, caption, GP_chat_id, bot)

        if high_count >= 2 and not currently_have_active_post_long_break:
            print(extracted_kline)
            position_type_long_break = "long"
            open_order_long_price_break_flag = current_highest
            currently_have_active_post_long_break = True
            ##### plot
            plot_time = convert_ms_to_utc_date_str(server_time)
            plot_ohlc(extracted_kline, plot_time, 60, current_highest + (current_highest * 0.1 / 100), None)
            image_path = 'ohlc_plot_60.png'
            caption = 'Long: current Kline and potential opening position price.'
            await send_image_message(image_path, caption, GP_chat_id, bot)





        """
            every time check if the following condition been triggered
                1. TP or SL triggerd
                2. trend condition triggerd 
        """
        if TP_order_list:

            TP_status = get_order_status(exchange, futures_symbol, TP_order_list)
            # if TP_status['status'] == "FILLED" or TP_status['status'] == "CANCELED":
            if TP_status['status'] == "FILLED":
                print("TP Triggered")
                await send_text_message(msg="TP Triggered",
                                        chat_id=PS_chat_id)
                _, server_time = await sync_server_time(exchange)
                await cancel_ALL_orders(exchange, futures_symbol, server_time)
                # cancel_orders(exchange, futures_symbol, SL_order_list, server_time)
                make_everthing_None = True




        if SL_order_list:

            SL_status = get_order_status(exchange, futures_symbol, SL_order_list)
            # if SL_status['status'] == "FILLED" or SL_status['status'] == "CANCELED":
            if SL_status['status'] == "FILLED":
                print("SL Triggered")
                await send_text_message(msg="SL Triggered",
                                        chat_id=PS_chat_id)

                _, server_time = await sync_server_time(exchange)
                await cancel_ALL_orders(exchange, futures_symbol, server_time)
                # cancel_orders(exchange, futures_symbol, TP_order_list, server_time)
                make_everthing_None = True

                not_place_order_for_next_few_min = True
                SL_number += 1


        if make_everthing_None:
            _, server_time = await sync_server_time(exchange)
            await cancel_ALL_orders(exchange, futures_symbol, server_time)

            ############# average position price


            currently_have_active_post_break = None
            ################ position type
            position_type = None
            position_type_short_break = None
            position_type_long_break = None

            position_type_T_1 = None
            position_type_T_2 = None
            position_type_4 = None

            open_order_long_price = None
            open_order_short_price = None


            open_order_long_price_break_flag = None
            open_order_short_price_break_flag = None

            ############### variable which same position pirce


            open_order_price_trend_1 = None
            open_order_price_trend_2 = None

            open_order_price_4 = None

            ############## indicator for trend break and 4 position; prevent place duplicate order
            currently_have_active_post_short_break = True
            currently_have_active_post_long_break = True
            currently_have_active_position_break = None
            currently_have_active_position_trend_1 = None
            currently_have_active_position_trend_2 = None
            currently_have_active_position_4 = None

            currently_have_active_order = None

            ################### tp and sl price; there should only be one tp and sl;
            TP_price = None
            SL_price = None
            ##################### variable to make none
            make_everthing_None = False
            current_profit_TP_SL = None
            current_profit_Dynamic = None
            balance_TP_SL = None
            balance_Dynamic = None
            position_quantity = None
            TP_order_list = None
            SL_order_list = None



            Tele_start_place_order = False
            order_book = []

        """
            1. for a single k line, if its up or down percentage >= 0.9%; place order with market price at the price
                when >= 0.9% condition meet.
        """
        if not currently_have_active_position_break and not not_place_order_for_next_few_min and SL_number <= 6 and Tele_start_place_order:
            new_open = extracted_kline[0][-1]
            new_close = extracted_kline[3][-1]
            new_high = extracted_kline[1][-1]
            new_low = extracted_kline[2][-1]
            # so if open price is bigger than close price it is a red k line

            if position_type_long_break == "long":
                open_order_price_break = open_order_long_price_break_flag + open_order_long_price_break_flag * 0.1 / 100
                if new_close >= open_order_price_break:
                    position_type_short_break = None
                    offset, server_time = await sync_server_time(exchange)
                    balance = await get_current_balance(exchange, server_time)
                    # open_order_price_trend_1 = min(TEST_OPEN_PRICE, new_close)
                    open_order_price_trend_1 = new_close
                    currently_have_active_position_break = True
                    target_SL = new_low
                    target_SL = target_SL - target_SL * 0.1 / 100
                    # calculate SL ratio
                    SL_ratio = (target_SL - open_order_price_trend_1) / open_order_price_trend_1
                    SL_ratio = abs(SL_ratio)

                    # I adjust SL to lower value to reduce SL loss every trade

                    # target_SL = open_order_price_trend_1 * (1 + SL_moving_ratio)

                    target_TP = open_order_price_trend_1 * (1 + (SL_ratio * TP_SL_ratio))

                    TP_price = round(target_TP)
                    SL_price = round(target_SL)

                    main_order_side = "BUY"
                    TP_order_side = "SELL"
                    SL_order_side = "SELL"
                    main_order_type = "MARKET"
                    TP_order_type = "TAKE_PROFIT_MARKET"
                    SL_order_type = "STOP_MARKET"



                    current_position_size = loss_per_trade / SL_ratio

                    if current_position_size > max_position_size:
                        print("margin insufficient")
                        current_position_size = max_position_size

                    position_quantity = round(current_position_size / open_order_price_trend_1, 3)

                    _, server_time = await sync_server_time(exchange)
                    # cancel_ALL_orders(exchange, futures_symbol, server_time)
                    main_order_id = await post_futures_order(exchange, futures_symbol, main_order_side, main_order_type,
                                                       position_quantity, server_time, None)
                    TP_order_id = await post_futures_order(exchange, futures_symbol, TP_order_side, TP_order_type, None,
                                                     server_time, TP_price)
                    while not TP_order_id:
                        _, server_time = await sync_server_time(exchange)
                        i = 2
                        TP_price = round(open_order_price_trend_1 * (1 + (i * SL_ratio * TP_SL_ratio)))
                        TP_order_id = await post_futures_order(exchange, futures_symbol, TP_order_side, TP_order_type,
                                                         None,
                                                         server_time, TP_price)
                        i += 1

                    SL_order_id = await post_futures_order(exchange, futures_symbol, SL_order_side, SL_order_type, None,
                                                     server_time, SL_price)
                    while not SL_order_id:
                        _, server_time = await sync_server_time(exchange)
                        i = 2
                        SL_price = round(target_SL - i * target_SL * 0.1 / 100)
                        SL_order_id = await post_futures_order(exchange, futures_symbol, SL_order_side, SL_order_type,
                                                         None,
                                                         server_time, SL_price)
                        i += 1

                    long_order = {"order_type": "long", "order_price": new_close, "order_quantity": position_quantity,
                                   "order_status": "open", "Balance": balance}
                    # append the short order to order book text file
                    with open('order_book_break.txt', 'a') as f:
                        f.write(f"Long Order: {long_order}\n")
                    order_book.append(long_order)
                    await send_text_message(msg="### opened long position! ###",
                                            chat_id=GP_chat_id)
                    TP_order_list = (TP_order_id)
                    SL_order_list = (SL_order_id)







            # else the k line is green line
            if position_type_short_break == "short":
                # 运行的趋势中 close 就是最新的值
                open_order_price_break = open_order_short_price_break_flag - open_order_short_price_break_flag * 0.1 / 100
                if new_close <= open_order_price_break:
                    position_type_long_break = None
                    offset, server_time = await sync_server_time(exchange)
                    balance = await get_current_balance(exchange, server_time)

                    # open_order_price_trend_1 = min(TEST_OPEN_PRICE, new_close)
                    open_order_price_trend_1 = new_close
                    currently_have_active_position_break = True
                    target_SL = new_high
                    target_SL = target_SL + target_SL * 0.1 / 100
                    # calculate SL ratio
                    SL_ratio = (target_SL - open_order_price_trend_1) / open_order_price_trend_1
                    SL_ratio = abs(SL_ratio)

                    # I adjust SL to lower value to reduce SL loss every trade

                    # target_SL = open_order_price_trend_1 * (1 + SL_moving_ratio)

                    target_TP = open_order_price_trend_1 * (1 - (SL_ratio * TP_SL_ratio))

                    TP_price = round(target_TP)
                    SL_price = round(target_SL)

                    main_order_side = "SELL"
                    TP_order_side = "BUY"
                    SL_order_side = "BUY"
                    main_order_type = "MARKET"
                    TP_order_type = "TAKE_PROFIT_MARKET"
                    SL_order_type = "STOP_MARKET"



                    current_position_size = loss_per_trade / SL_ratio

                    if current_position_size > max_position_size:
                        print("margin insufficient")
                        current_position_size = max_position_size

                    position_quantity = round(current_position_size / open_order_price_trend_1, 3)

                    _, server_time = await sync_server_time(exchange)
                    # cancel_ALL_orders(exchange, futures_symbol, server_time)
                    main_order_id = await post_futures_order(exchange, futures_symbol, main_order_side, main_order_type,
                                                       position_quantity, server_time, None)
                    TP_order_id = await post_futures_order(exchange, futures_symbol, TP_order_side, TP_order_type, None,
                                                     server_time, TP_price)
                    while not TP_order_id:
                        _, server_time = await sync_server_time(exchange)
                        i = 2
                        TP_price = round(open_order_price_trend_1 * (1 - (i * SL_ratio * TP_SL_ratio)))
                        TP_order_id = await post_futures_order(exchange, futures_symbol, TP_order_side, TP_order_type,
                                                         None,
                                                         server_time, TP_price)
                        i += 1

                    SL_order_id = await post_futures_order(exchange, futures_symbol, SL_order_side, SL_order_type, None,
                                                     server_time, SL_price)
                    while not SL_order_id:
                        _, server_time = await sync_server_time(exchange)
                        i = 2
                        SL_price = round(target_SL + i * target_SL * 0.1 / 100)
                        SL_order_id = await post_futures_order(exchange, futures_symbol, SL_order_side, SL_order_type,
                                                         None,
                                                         server_time, SL_price)
                        i += 1

                    short_order = {"order_type": "short", "order_price": new_close, "order_quantity": position_quantity,
                                   "order_status": "open", "Balance": balance}
                    # append the short order to order book text file
                    with open('order_book_break.txt', 'a') as f:
                        f.write(f"Short Order: {short_order}\n")
                    order_book.append(short_order)
                    await send_text_message(msg="### opened short position! ###",
                                            chat_id=GP_chat_id)
                    TP_order_list = (TP_order_id)
                    SL_order_list = (SL_order_id)






def convert_ms_to_utc_date_str(current_1m_kline_time):
    # Convert string to integer
    timestamp_ms = int(current_1m_kline_time)

    # Convert milliseconds to seconds
    timestamp_s = timestamp_ms / 1000.0

    # Create a datetime object from the timestamp (in seconds)
    date_time = datetime.datetime.utcfromtimestamp(timestamp_s)

    # Format the datetime object to the desired string format
    formatted_date_str = date_time.strftime('%Y-%m-%d %H:%M:%S')

    return formatted_date_str


async def main():
    # Initialize CCXT for Binance Futures
    exchange = ccxt.binance({
        'apiKey': '0irxPyL1G7cy3cJoIOyNegZlWemRFRetMgeZ046Iud6oCQmvoLFVkYeDry3O82t5',
        'secret': 'URQsMETBGblyNKWNnlqsRtVQHyBgJ80iqATf9DPPlhB9K8yGBpj3xpOS7TOZBByU',
        'enableRateLimit': True  # Recommended by ccxt when making frequent requests
    })

    # Define the trading pair and contract type
    futures_symbol = "BTCUSDT"
    contract_type = 'PERPETUAL'
    leverage = 20
    investment_percent = 0.30
    maker_fee = 0.02  # 0.02% maker fee
    taker_fee = 0.05  # 0.05% taker fee

    threshold_percentage = 0.15

    TP_SL_ratio = 11
    loss_per_trade = 5
    # k1_ratio = 0.008
    k1_ratio = 0.000328 * 2
    k2_ratio = 0.015

    SL_moving_ratio = 0.003

    default_window_size = 48


    update_queue = asyncio.Queue()
    command_queue = asyncio.Queue()
    updater = Updater(bot=bot, update_queue=update_queue)

    async with updater:
        updater_task = asyncio.create_task(updater.start_polling())
        processing_task = asyncio.create_task(process_updates(update_queue, bot, command_queue, PS_chat_id))
        trading_task = asyncio.create_task(backtest_strategy(exchange, leverage=leverage, investment_percent=investment_percent,
                                  maker_fee_percentage=maker_fee,
                                  taker_fee_percentage=taker_fee, futures_symbol=futures_symbol,
                                  contract_type=contract_type,
                                  threshold_percentage=threshold_percentage, TP_SL_ratio=TP_SL_ratio,
                                  loss_per_trade=loss_per_trade,
                                  k1_ratio=k1_ratio, k2_ratio=k2_ratio, SL_moving_ratio=SL_moving_ratio,
                                  default_window=default_window_size, command_queue=command_queue,
                                  ))

        try:
            await asyncio.gather(updater_task, processing_task, trading_task)
        except asyncio.CancelledError:
            print("Tasks were cancelled.")
        finally:
            # Ensure this code runs after the tasks are cancelled
            print("Executing code after the tasks...")


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, loop.stop)
    try:
        loop.run_until_complete(main())
    finally:
        print("Shutting down...")
