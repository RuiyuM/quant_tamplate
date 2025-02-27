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
import math
import ta
from Feature import *
from utils import *
import LorentzianModel
import GridLine as GL

from telegram.error import TelegramError
import asyncio
import signal
from telegram import Bot, Update
from telegram.ext import Updater
from DataCatch import dataCatch
import sys
from receieve_and_send import process_updates
from TakerEnergy import TakerEnergy

bot_token = ''
PS_chat_id = ''
GP_chat_id = ''
bot = Bot(token=bot_token)
##### Telegram command
Tele_com_auto_open_long_position = "open_auto_long"
Tele_com_auto_open_short_position = "open_auto_short"
Tele_com_auto_close_long_position = "close_auto_long"
Tele_com_auto_close_short_position = "close_auto_short"
Tele_com_close_position = "close"
Tele_com_get_balance = "balance"
Tele_com_make_all_none = "none"
Tele_com_open_volume_filter = "open_volume_filter"
Tele_com_close_volume_filter = "close_volume_filter"


async def backtest_strategy(exchange, leverage,
                            futures_symbol,
                            contract_type,
                            threshold_percentage, TP_SL_ratio,
                            loss_per_trade,
                            command_queue,
                            timeframe_merge,
                            use_Volume_filter,
                            shortlen,
                            longlen,
                            max_position_quantity_grid
                            ):
    timeframe_merge = int(timeframe_merge[:-1])
    starttime = time.time()
    timeframe = '1m'
    setting_leverage = await set_leverage(exchange, futures_symbol, leverage)
    if setting_leverage:
        await send_text_message(bot=bot, msg=f"### Set leverage to {leverage} ###",
                                chat_id=PS_chat_id)
    Tele_adjust_loss_per_trade = loss_per_trade
    order_book = []
    long_order_confirm = []
    long_order_confirm2 = False
    short_order_confirm = []
    short_order_confirm2 = False
    BullBearFlip_confirm = []
    TP_price = None
    SL_price = None
    TP_order_list = None
    SL_order_list = None
    replaceSL = True
    isBullish = False
    isBearish = False
    closePosition = True
    initial_entry_price = None
    Signal_price = None
    universal_depth_data = []

    maxProfit = 0
    Profitdown = 0.95
    sameKlineOpenPosition = True

    make_everthing_None = False
    Tele_auto_place_long_order = False
    Tele_auto_place_short_order = False
    Tele_auto_close_long_order = False
    Tele_auto_close_short_order = False
    Tele_close_all = False

    lastOpenLong = False
    lastOpenShort = False
    lastCloseLong = False
    lastCloseShort = False
    ReverseOrder = False    
    stopsend = True
    PositionOIstatus = None
    OpenInterestFirst = await get_open_interest(exchange, futures_symbol)
    takerEnergy = TakerEnergy(exchange, futures_symbol)
    lr_train_buffer = await dataCatch(timeframe, 64000, futures_symbol)
    check_dimensions_initial(lr_train_buffer, 64000)
    print("Start to initialize the Lorentz Model")
    real_time_buffer, start_index = find_and_buffer(lr_train_buffer, timeframe_merge)
    resampled_df = resample_ohlc(lr_train_buffer, start_index - 1, timeframe_merge, 3999)
    buffer_ohlc = buffer_to_ohlc(real_time_buffer)
    final_df = pd.concat([resampled_df, buffer_ohlc], ignore_index=True)

    use_adx_filter = False
    LRmodel = LorentzianModel.LorentzianTradingModel(final_df, use_adx_filter, use_Volume_filter)
    LRmodel.useRegimeFilter = True
    LRmodel.Volume_shortlen = shortlen
    LRmodel.Volume_longlen = longlen
    endtime = time.time()
    initTime = endtime - starttime
    # convert initTime to minutes
    initTime = initTime / 60
    print("Initialize the Lorentz Model successfully")

    compensate_buffer = await dataCatch(timeframe, math.ceil(initTime) + 1, futures_symbol)

    offset, server_time = await sync_server_time(exchange)
    LastTime = is_first_second_of_minute(server_time, timeframe_merge)
    # print("Last", LastTime)
    Kline_first_second_of_minute = False
    for ind, row in compensate_buffer.iterrows():
        if row['timestamp'] == real_time_buffer.iloc[-1]['timestamp']:
            real_time_buffer.iloc[-1] = row
            break

    # lr_train_buffer = lr_train_buffer.iloc[2000:]
    final_df = final_df.iloc[2000:]
    for i in range(ind + 1, len(compensate_buffer)):
        # LRmodel.y_train_buffer = lr_train_buffer
        # LRmodel.process_row()
        if len(real_time_buffer) < timeframe_merge:
            real_time_buffer = pd.concat([real_time_buffer, compensate_buffer.iloc[i].to_frame().T], ignore_index=True)
        else:
            buffer_ohlc = buffer_to_ohlc(real_time_buffer)
            final_df.iloc[-1] = buffer_ohlc.iloc[-1]
            LRmodel.y_train_buffer = final_df
            LRmodel.process_row()
            if i < len(compensate_buffer):
                real_time_buffer = compensate_buffer.iloc[i:]
            break
    endtime2 = time.time()
    compensaTime = endtime2 - endtime

    while True:

        try:
            # command is (str) and sender_id is (int)
            command, sender_id = await asyncio.wait_for(command_queue.get(), timeout=1)
            if command and sender_id:
                if sender_id == int(PS_chat_id):
                    print(f"Received command: {command}")

                    if command.isdigit():
                        Tele_adjust_loss_per_trade = int(command)

                        await send_text_message(bot=bot,
                                                msg=f"### adjust loss per trade to {Tele_adjust_loss_per_trade}$ ###",
                                                chat_id=PS_chat_id)
                    elif command.lower() == Tele_com_auto_open_long_position.lower():
                        Tele_auto_place_long_order = True

                        await send_text_message(bot=bot, msg="### start auto open long position only! ###",
                                                chat_id=PS_chat_id)

                    elif command.lower() == Tele_com_auto_open_short_position.lower():
                        Tele_auto_place_short_order = True

                        await send_text_message(bot=bot, msg="### start auto open short position only! ###",
                                                chat_id=PS_chat_id)
                    elif command.lower() == Tele_com_auto_close_long_position.lower():
                        Tele_auto_close_long_order = True

                        await send_text_message(bot=bot, msg="### start auto close long position ###",
                                                chat_id=PS_chat_id)
                    elif command.lower() == Tele_com_auto_close_short_position.lower():
                        Tele_auto_close_short_order = True

                        await send_text_message(bot=bot, msg="### start auto close short position ###",
                                                chat_id=PS_chat_id)

                    elif command.lower() == Tele_com_close_position.lower():
                        Tele_close_all = True

                    elif command.lower() == Tele_com_get_balance.lower():
                        await send_text_message(bot=bot, msg=f"### current balance: {balance} ###",
                                                chat_id=GP_chat_id)
                    elif command.lower() == Tele_com_make_all_none.lower():
                        make_everthing_None = True
                        await send_text_message(bot=bot, msg=f"### make all variables to its initial states ###",
                                                chat_id=PS_chat_id)
                    elif command.lower() == Tele_com_open_volume_filter.lower():
                        LRmodel.useVolume_Osc = True
                        await send_text_message(bot=bot, msg=f"### open volume filter ###",
                                                chat_id=PS_chat_id)
                    elif command.lower() == Tele_com_close_volume_filter.lower():
                        LRmodel.useVolume_Osc = False
                        await send_text_message(bot=bot, msg=f"### close volume filter ###",
                                                chat_id=PS_chat_id)

            # Process the command and adjust the strategy accordingly
            command_queue.task_done()
        except asyncio.TimeoutError:
            pass

        ############# close all ############################
        if Tele_close_all:
            Tele_close_all = False
            if order_book:
                await close_all_position_at_order_book(exchange, order_book, new_close, balance, futures_symbol, bot,
                                                       PS_chat_id)
                order_book = []
                _, server_time = await sync_server_time(exchange)
                await cancel_ALL_orders(exchange, futures_symbol, server_time)
                initial_entry_price = None
            else:
                await send_text_message(bot=bot, msg="### Order book is empty, nothing to close ###",
                                        chat_id=PS_chat_id)

        offset, server_time = await sync_server_time(exchange)
        balance = await get_current_balance(exchange, server_time)
        max_position_size = balance * (leverage - 10)
        NewTime = is_first_second_of_minute(server_time, timeframe_merge)
        if NewTime != LastTime:
            Kline_first_second_of_minute = True
        else:
            Kline_first_second_of_minute = False
        LastTime = NewTime
        # 做一个容错就是如果这次fetch data 还是上一分钟的就re-fetch
        while True:

            new_data_1h = await dataCatch(timeframe, 5, futures_symbol)
            row_15 = new_data_1h.iloc[-1]

            if Kline_first_second_of_minute:
                while True:
                    if row_15['timestamp'] != real_time_buffer.iloc[-1]['timestamp']:
                        break
                    else:
                        sleep(0.5)
                        new_data_1h = await dataCatch(timeframe, 5, futures_symbol)
                        row_15 = new_data_1h.iloc[-1]


            if Kline_first_second_of_minute:

                real_time_buffer.iloc[-1] = new_data_1h.iloc[-2]
                real_time_buffer = filter_max_volume(real_time_buffer)
                buffer_ohlc = buffer_to_ohlc(real_time_buffer)
                final_df.iloc[-1] = buffer_ohlc.iloc[-1]
                LRmodel.y_train_buffer = final_df
                LRmodel.process_row()
                LRmodel.bars_held_count()

                if LRmodel.isBullish:
                    isBullish = True
                    isBearish = False
                elif LRmodel.isBearish:
                    isBearish = True
                    isBullish = False

                # real_time_buffer = row_15
                # remove all
                real_time_buffer = real_time_buffer.iloc[0:0]
                real_time_buffer = pd.concat([real_time_buffer, row_15.to_frame().T], ignore_index=True)
                buffer_ohlc = buffer_to_ohlc(real_time_buffer)
                final_df = pd.concat([final_df, buffer_ohlc], ignore_index=True)
                final_df = final_df.iloc[1:]
                sameKlineOpenPosition = True
                closePosition = True

                if order_book and ReverseOrder:
                    if order_book[0]['order_type'] == "long" and new_close > lastOpen:
                        if OpenInterestLatest > OpenInterestFirst:
                            KlineOIStatus = "UP"
                        else:
                            KlineOIStatus = "DOWN"

                        if PositionOIstatus != KlineOIStatus:
                            await close_all_position_at_order_book(exchange, order_book, new_close, balance,
                                                                futures_symbol, bot, PS_chat_id)
                            order_book = []
                            TP_order_list = None
                            SL_order_list = None
                            initial_entry_price = None
                    elif order_book[0]['order_type'] == "short" and new_close < lastOpen:
                        if OpenInterestLatest > OpenInterestFirst:
                            KlineOIStatus = "UP"
                        else:
                            KlineOIStatus = "DOWN"
                        
                        if PositionOIstatus != KlineOIStatus:
                            await close_all_position_at_order_book(exchange, order_book, new_close, balance,
                                                                futures_symbol, bot, PS_chat_id)
                            order_book = []
                            TP_order_list = None
                            SL_order_list = None
                            initial_entry_price = None

                OpenInterestFirst = await get_open_interest(exchange, futures_symbol)
                lastOpen = new_close


            else:
                # Replace the last row in local_data_15_window with row_15
                real_time_buffer = pd.concat([real_time_buffer, row_15.to_frame().T], ignore_index=True)
                real_time_buffer = filter_max_volume(real_time_buffer)
                buffer_ohlc = buffer_to_ohlc(real_time_buffer)
                final_df.iloc[-1] = buffer_ohlc.iloc[-1]

            break

        """
                    every time check if the following condition been triggered
                        1. TP or SL triggerd
                        2. trend condition triggerd 
                """
        if TP_order_list:

            TP_status = await get_order_status(exchange, futures_symbol, TP_order_list)
            # if TP_status['status'] == "FILLED" or TP_status['status'] == "CANCELED":
            if TP_status['status'] == "FILLED":
                print("TP Triggered")
                await send_text_message(bot=bot, msg="TP Triggered",
                                        chat_id=PS_chat_id)
                _, server_time = await sync_server_time(exchange)
                await cancel_ALL_orders(exchange, futures_symbol, server_time)
                order_book = []
                maxProfit = 0
                # cancel_orders(exchange, futures_symbol, SL_order_list, server_time)
                # make_everthing_None = True
                TP_order_list = None
                SL_order_list = None

        if SL_order_list:

            SL_status = await get_order_status(exchange, futures_symbol, SL_order_list)
            # if SL_status['status'] == "FILLED" or SL_status['status'] == "CANCELED":
            if SL_status['status'] == "FILLED":
                print("SL Triggered")
                await send_text_message(bot=bot, msg="SL Triggered",
                                        chat_id=PS_chat_id)

                _, server_time = await sync_server_time(exchange)
                await cancel_ALL_orders(exchange, futures_symbol, server_time)
                order_book = []
                maxProfit = 0

                # cancel_orders(exchange, futures_symbol, TP_order_list, server_time)
                TP_order_list = None
                SL_order_list = None

        if make_everthing_None:
            _, server_time = await sync_server_time(exchange)
            await cancel_ALL_orders(exchange, futures_symbol, server_time)
            make_everthing_None = False
            Tele_auto_place_long_order = False
            Tele_auto_place_short_order = False
            Tele_auto_close_long_order = False
            Tele_auto_close_short_order = False
            Tele_close_all = False
            LRmodel.useVolume_Osc = False

        
        date = convert_binance_server_time(server_time)
        bids, asks = await get_depth_info(exchange, futures_symbol)
        depth_result = analyze_order_depth(bids, asks)
        new_close = final_df.iloc[-1]['close']
        new_volume = final_df.iloc[-1]['volume']
        OpenInterestLatest = await get_open_interest(exchange, futures_symbol)
        LRmodel.y_train_buffer = final_df
        res = LRmodel.process_row()
        Filter_direaction = EMA_SMA_Filter(final_df, 200)
        if res[0]:
            long_order_confirm.append(res[0])
        else:
            long_order_confirm = []

        if res[1]:
            short_order_confirm.append(res[1])
        else:
            short_order_confirm = []

        if len(long_order_confirm) == 1 or len(short_order_confirm) == 1:
            Signal_price = new_close

        # with open('result15.txt', 'a') as f:
        #     f.write(str([date, res, list(LRmodel.Signal_array.values())[-1]]) + ' isBullish:' + str(
        #         LRmodel.isBullish) + ' isBearish:' + str(LRmodel.isBearish) + ' yhat1:' + str(
        #         list(LRmodel.yhat1_array.values())[-1]) + ' Barheld:' + str(LRmodel.bars_held) + '\n')

        depth_result['Date'] = str(date)
        date_obj = datetime.datetime.strptime(depth_result['Date'], '%Y-%m-%d %H:%M:%S')
        universal_depth_data.append((date_obj, depth_result['Bid/Ask Ratio']))
        if len(universal_depth_data) > 2400:
            universal_depth_data.pop(0)
        result_str = (f"Date: {depth_result['Date']}, "
                      f"Total Bids: {depth_result['Total Bids']}, "
                      f"Total Asks: {depth_result['Total Asks']}, "
                      f"Bid/Ask Ratio: {depth_result['Bid/Ask Ratio']}, "
                      f"Market Sentiment: {depth_result['Market Sentiment']}\n")

        with open('depth_results_BTC.txt', 'a') as f:
            f.write(result_str)

        yhat1 = list(LRmodel.yhat1_array.values())[-1]
        buffer_percent = 0.0015
        long_threshold = yhat1 * (1 - buffer_percent)  # Decrease yhat1 by 0.15% for long
        short_threshold = yhat1 * (1 + buffer_percent)  # Increase yhat1 by 0.15% for short
        if order_book and closePosition and not ReverseOrder:
            if order_book[0]['order_type'] == "long" and new_close < long_threshold:
                await close_all_position_at_order_book(exchange, order_book, new_close, balance,
                                                       futures_symbol, bot, PS_chat_id)
                order_book = []
                TP_order_list = None
                SL_order_list = None
                initial_entry_price = None
            elif order_book[0]['order_type'] == "short" and new_close > short_threshold:
                await close_all_position_at_order_book(exchange, order_book, new_close, balance,
                                                       futures_symbol, bot, PS_chat_id)
                order_book = []
                TP_order_list = None
                SL_order_list = None
                initial_entry_price = None
        if order_book and initial_entry_price:
            _, server_time = await sync_server_time(exchange)
            position_status = await get_position_status(exchange, futures_symbol, server_time)
            net_profit = float(position_status[0]['unRealizedProfit'])
            current_position_size = abs(float(position_status[0]["positionAmt"]))
            order_book[0]['order_quantity'] = current_position_size
            if order_book[0]['order_type'] == "short":
                trading_signal = grid221_biased.get_trading_signal(new_close)
            else:
                trading_signal = grid221_biased.get_trading_signal(new_close)
            trading_signal_case_2 = grid221_unbiased.get_trading_signal(new_close)
            if order_book[0]['order_type'] == "long":
                if net_profit > float(current_position_size) * float(position_status[0]['entryPrice']) * 0.001:
                    if trading_signal == 'buy':
                        print("buy")

                    elif trading_signal == 'sell':
                        print("sell")

                else:
                    if trading_signal_case_2 == 'buy':
                        if current_position_size != 0.0:
                            main_order_side = "SELL"

                            main_order_type = "MARKET"

                            reduce_only = True
                            position_quantity = position_quantity_list[0]
                            position_quantity_list.pop(0)
                            _, server_time = await sync_server_time(exchange)
                            main_order_id = await post_futures_order(exchange, futures_symbol, main_order_side,
                                                                     main_order_type,
                                                                     position_quantity, server_time, PS_chat_id, bot,
                                                                     None)
                            position_status = await get_position_status(exchange, futures_symbol, server_time)
                            current_position_size = abs(float(position_status[0]["positionAmt"]))
                            order_book[0]['order_quantity'] = current_position_size

                    elif trading_signal_case_2 == 'sell':
                        print("sell")


            if order_book[0]['order_type'] == "short":
                if net_profit > float(current_position_size) * float(position_status[0]['entryPrice']) * 0.001:
                    if trading_signal == 'buy':
                        print("buy")

                    elif trading_signal == 'sell':
                        print("sell")

                else:
                    if trading_signal_case_2 == 'buy':
                        print("sell")


                    elif trading_signal_case_2 == 'sell':
                        if current_position_size != 0.0:
                            main_order_side = "BUY"

                            main_order_type = "MARKET"

                            reduce_only = True
                            position_quantity = position_quantity_list[0]
                            position_quantity_list.pop(0)
                            _, server_time = await sync_server_time(exchange)
                            main_order_id = await post_futures_order(exchange, futures_symbol, main_order_side,
                                                                     main_order_type,
                                                                     position_quantity, server_time, PS_chat_id, bot,
                                                                     None)
                            position_status = await get_position_status(exchange, futures_symbol, server_time)
                            current_position_size = abs(float(position_status[0]["positionAmt"]))
                            order_book[0]['order_quantity'] = current_position_size



        if res[0]:
            if res[0] != lastOpenLong:
                await send_text_message(bot=bot, msg=f"### Open long signal for {timeframe_merge} period! ###",
                                        chat_id=GP_chat_id)
                
                stopsend = True

                offset, server_time = await sync_server_time(exchange)
                dataTime = datetime.datetime.fromtimestamp(server_time/1000).strftime('%Y-%m-%d %H:%M:%S')
                with open('BTCSignalTime.csv', mode='a') as file:
                    writer = csv.writer(file)
                    writer.writerow([dataTime])
            if Tele_auto_place_long_order and (len(long_order_confirm) > 5) and sameKlineOpenPosition and new_close > yhat1:# and Filter_direaction == True:
                
                if order_book:
                    if order_book[0]['order_type'] == "short":
                        await close_all_position_at_order_book(exchange, order_book, new_close, balance,
                                                               futures_symbol, bot, PS_chat_id)
                        order_book = []
                        _, server_time = await sync_server_time(exchange)
                        await cancel_ALL_orders(exchange, futures_symbol, server_time)
                        TP_order_list = None
                        SL_order_list = None
                        initial_entry_price = None
                    elif order_book[0]['order_type'] == "long":
                        await close_all_position_at_order_book(exchange, order_book, new_close, balance,
                                                               futures_symbol, bot, PS_chat_id)
                        order_book = []
                        _, server_time = await sync_server_time(exchange)
                        await cancel_ALL_orders(exchange, futures_symbol, server_time)
                        TP_order_list = None
                        SL_order_list = None
                        initial_entry_price = None
                else:
                    open_signal_variance = False
                    Long_short_info = await get_top_long_short_ratio(exchange, futures_symbol)
                    toplongshortratio = float(Long_short_info[-1]['longShortRatio'])
                    list_of_ratio = [float(item['longShortRatio']) * 100 for item in Long_short_info]
                    list_of_ratio = sorted(list_of_ratio)
                    new_differences = [list_of_ratio[i] - list_of_ratio[i + 1] for i in range(len(list_of_ratio) - 1)]

                    # 计算平均差值
                    new_average_difference = abs(sum(new_differences) / len(new_differences))

                    df = pd.DataFrame(universal_depth_data, columns=["Date", "Bid/Ask Ratio"])

                    # Set the 'Date' column as the index for easier resampling
                    df.set_index('Date', inplace=True)

                    # Resample data by the specified interval (5 minutes in this case) and calculate min, max, and average Bid/Ask Ratio
                    grouped_depth_ratio = df.resample('1T').agg(
                        Min_Ratio=('Bid/Ask Ratio', 'min'),
                        Max_Ratio=('Bid/Ask Ratio', 'max'),
                        Avg_Ratio=('Bid/Ask Ratio', 'mean')
                    ).dropna()
                    grouped_depth_ratio = grouped_depth_ratio[-6:-1]
                    avg_ratio_list = extract_depth_related_ratios(grouped_depth_ratio["Avg_Ratio"])
                    max_ratio_list = extract_depth_related_ratios(grouped_depth_ratio["Max_Ratio"])
                    min_ratio_list = extract_depth_related_ratios(grouped_depth_ratio["Min_Ratio"])

                    # Check the signal
                    close_signal_R = order_book_close_signal(avg_ratio_list, max_ratio_list, min_ratio_list)

                    if close_signal_R < 0.99:
                        open_signal_variance = True
                    takerEnergyFlag, OIdropFlag = takerEnergy.get_taker_energy()
                    # new_average_difference = new_average_difference * modified_sigmoid_smooth(toplongshortratio)
                    # OpenInterestCurrent = await get_open_interest(exchange, futures_symbol)
                    if stopsend:
                        await send_text_message(bot=bot, msg=f"### LongShortRatioDiff: {new_average_difference}, TakerEnergy: {takerEnergyFlag}, OIdrop: {OIdropFlag}, order_book: {open_signal_variance}#{close_signal_R} ###",
                                                chat_id=GP_chat_id)
                        stopsend = False

                    # if (not takerEnergy) and OIdropFlag and not Filter_direaction and not open_signal_variance:
                    #
                    #     offset, server_time = await sync_server_time(exchange)
                    #     balance = await get_current_balance(exchange, server_time)
                    #
                    #     main_order_side = "SELL"
                    #     main_order_type = "MARKET"
                    #
                    #     position_quantity = 0.004  # Tele_adjust_loss_per_trade*0.003 # max(position_quantity, 0.003)
                    #     position_quantity_list = [0.002, 0.002]
                    #     _, server_time = await sync_server_time(exchange)
                    #     await cancel_ALL_orders(exchange, futures_symbol, server_time)
                    #     main_order_id = await post_futures_order(exchange, futures_symbol, main_order_side, main_order_type,
                    #                                              position_quantity, server_time, PS_chat_id, bot, None)
                    #
                    #     OpenInterestCurrent = await get_open_interest(exchange, futures_symbol)
                    #     if OpenInterestCurrent > OpenInterestFirst:
                    #         PositionOIstatus = "UP"
                    #     else:
                    #         PositionOIstatus = "DOWN"
                    #
                    #     position_status = await get_position_status(exchange, futures_symbol, server_time)
                    #     if initial_entry_price is None:
                    #         initial_entry_price = float(position_status[0]['entryPrice'])
                    #         grid_size = round(initial_entry_price * 221.87 / 60000, 4)
                    #         grid221_biased = GL.GridLine(initial_entry_price, grid_size, 1000, main_order_side)
                    #         grid221_unbiased = GL.GridLine(initial_entry_price, grid_size, 1000, main_order_side)
                    #
                    #
                    #
                    #     short_order = {"order_type": "short", "order_price": new_close, "order_quantity": position_quantity,
                    #                    "order_status": "open", "Balance": balance}
                    #     # append the short order to order book text file
                    #     with open('order_book_break.txt', 'a') as f:
                    #         f.write(f"Short Order: {short_order}\n")
                    #     order_book.append(short_order)
                    #     await send_text_message(bot=bot, msg="### opened short position! ###",
                    #                             chat_id=PS_chat_id)
                    #     replaceSL = True
                    #     ReverseOrder = True
                    #     sameKlineOpenPosition = False
                    #     closePosition = False
                    if (not takerEnergyFlag) or OIdropFlag:
                        sameKlineOpenPosition = False
                    # if close_signal_R > 0.99:
                    #     sameKlineOpenPosition = False

                    if new_average_difference >= 0.7 and takerEnergyFlag and not OIdropFlag and open_signal_variance:
                        offset, server_time = await sync_server_time(exchange)
                        balance = await get_current_balance(exchange, server_time)

                        open_order_price_trend_1 = new_close

                        target_SL = min(final_df.iloc[-1]['low'], final_df.iloc[-2]['low'])
                        target_SL = target_SL - target_SL * threshold_percentage / 100
                        # calculate SL ratio
                        SL_ratio = (target_SL - open_order_price_trend_1) / open_order_price_trend_1
                        SL_ratio = abs(SL_ratio)

                        # I adjust SL to lower value to reduce SL loss every trade

                        # target_SL = open_order_price_trend_1 * (1 + SL_moving_ratio)

                        target_TP = open_order_price_trend_1 * (1 + 0.0073)

                        TP_price = round(target_TP)
                        SL_price = round(target_SL)

                        main_order_side = "BUY"
                        TP_order_side = "SELL"
                        SL_order_side = "SELL"
                        main_order_type = "MARKET"
                        TP_order_type = "TAKE_PROFIT_MARKET"
                        SL_order_type = "STOP_MARKET"

                        current_position_size = Tele_adjust_loss_per_trade / SL_ratio

                        if current_position_size > max_position_size:
                            print("margin insufficient")
                            current_position_size = max_position_size

                        position_quantity = round(current_position_size / open_order_price_trend_1, 3)
                        position_quantity = 0.004  # Tele_adjust_loss_per_trade*0.003 # max(position_quantity, 0.003)
                        position_quantity_list = [0.002, 0.002]
                        _, server_time = await sync_server_time(exchange)
                        await cancel_ALL_orders(exchange, futures_symbol, server_time)
                        main_order_id = await post_futures_order(exchange, futures_symbol, main_order_side, main_order_type,
                                                                 position_quantity, server_time, PS_chat_id, bot, None)
                        position_status = await get_position_status(exchange, futures_symbol, server_time)
                        if initial_entry_price is None:
                            initial_entry_price = float(position_status[0]['entryPrice'])
                            grid_size = round(initial_entry_price * 221.87 / 60000, 4)
                            grid221_biased = GL.GridLine(initial_entry_price, grid_size, 1000, main_order_side)
                            grid221_unbiased = GL.GridLine(initial_entry_price, grid_size, 1000, main_order_side)

                        initial_entry_price = None

                        if Filter_direaction == False:
                            TP_order_id = await post_futures_order(exchange, futures_symbol, TP_order_side, TP_order_type, None,
                                                                   server_time, PS_chat_id, bot, TP_price)
                            TP_order_list = (TP_order_id)
                            # while not TP_order_id:
                            #     _, server_time = await sync_server_time(exchange)
                            #     i = 2
                            #     TP_price = round(open_order_price_trend_1 * (1 + (i * SL_ratio * TP_SL_ratio)))
                            #     TP_order_id = await post_futures_order(exchange, futures_symbol, TP_order_side, TP_order_type,
                            #                                            None,
                            #                                            server_time, PS_chat_id, bot, TP_price)
                        #     i += 1

                        # SL_order_id = await post_futures_order(exchange, futures_symbol, SL_order_side, SL_order_type, None,
                        #                                        server_time, PS_chat_id, bot, SL_price)
                        # while not SL_order_id:
                        #     _, server_time = await sync_server_time(exchange)
                        #     i = 2
                        #     SL_price = round(target_SL - i * target_SL * threshold_percentage / 100)
                        #     SL_order_id = await post_futures_order(exchange, futures_symbol, SL_order_side, SL_order_type,
                        #                                            None,
                        #                                            server_time, PS_chat_id, bot, SL_price)
                        #     i += 1

                        long_order = {"order_type": "long", "order_price": new_close, "order_quantity": position_quantity,
                                      "order_status": "open", "Balance": balance}
                        # append the short order to order book text file
                        with open('order_book_break.txt', 'a') as f:
                            f.write(f"Long Order: {long_order}\n")
                        order_book.append(long_order)
                        await send_text_message(bot=bot, msg="### opened long position! ###",
                                                chat_id=PS_chat_id)
                        # TP_order_list = (TP_order_id)
                        # SL_order_list = (SL_order_id)
                        replaceSL = True
                        ReverseOrder = False
                        sameKlineOpenPosition = False
                        closePosition = False

        elif res[1]:
            if res[1] != lastOpenShort:
                await send_text_message(bot=bot, msg=f"### Open short signal for {timeframe_merge} period! ###",
                                        chat_id=GP_chat_id)
                stopsend = True
                _, server_time = await sync_server_time(exchange)
                dataTime = datetime.datetime.fromtimestamp(server_time/1000).strftime('%Y-%m-%d %H:%M:%S')
                with open('BTCSignalTime.csv', mode='a') as file:
                    writer = csv.writer(file)
                    writer.writerow([dataTime])
            if Tele_auto_place_short_order and (len(short_order_confirm) > 5) and sameKlineOpenPosition and new_close < yhat1:# and Filter_direaction == False:
                
                if order_book:
                    if order_book[0]['order_type'] == "long":
                        await close_all_position_at_order_book(exchange, order_book, new_close, balance,
                                                               futures_symbol, bot, PS_chat_id)
                        order_book = []
                        _, server_time = await sync_server_time(exchange)
                        await cancel_ALL_orders(exchange, futures_symbol, server_time)
                        TP_order_list = None
                        SL_order_list = None
                        initial_entry_price = None
                    elif order_book[0]['order_type'] == "short":
                        await close_all_position_at_order_book(exchange, order_book, new_close, balance,
                                                               futures_symbol, bot, PS_chat_id)
                        order_book = []
                        _, server_time = await sync_server_time(exchange)
                        await cancel_ALL_orders(exchange, futures_symbol, server_time)
                        TP_order_list = None
                        SL_order_list = None
                        initial_entry_price = None
                else:
                    open_signal_variance = False
                    Long_short_info = await get_top_long_short_ratio(exchange, futures_symbol)
                    toplongshortratio = float(Long_short_info[-1]['longShortRatio'])
                    list_of_ratio = [float(item['longShortRatio']) * 100 for item in Long_short_info]
                    list_of_ratio = sorted(list_of_ratio)
                    new_differences = [list_of_ratio[i] - list_of_ratio[i + 1] for i in range(len(list_of_ratio) - 1)]

                    # 计算平均差值
                    new_average_difference = abs(sum(new_differences) / len(new_differences))

                    df = pd.DataFrame(universal_depth_data, columns=["Date", "Bid/Ask Ratio"])

                    # Set the 'Date' column as the index for easier resampling
                    df.set_index('Date', inplace=True)

                    # Resample data by the specified interval (5 minutes in this case) and calculate min, max, and average Bid/Ask Ratio
                    grouped_depth_ratio = df.resample('1T').agg(
                        Min_Ratio=('Bid/Ask Ratio', 'min'),
                        Max_Ratio=('Bid/Ask Ratio', 'max'),
                        Avg_Ratio=('Bid/Ask Ratio', 'mean')
                    ).dropna()
                    grouped_depth_ratio = grouped_depth_ratio[-6:-1]
                    avg_ratio_list = extract_depth_related_ratios(grouped_depth_ratio["Avg_Ratio"])
                    max_ratio_list = extract_depth_related_ratios(grouped_depth_ratio["Max_Ratio"])
                    min_ratio_list = extract_depth_related_ratios(grouped_depth_ratio["Min_Ratio"])

                    # Check the signal
                    close_signal_R = order_book_close_signal(avg_ratio_list, max_ratio_list, min_ratio_list)

                    if close_signal_R > 1.1:
                        open_signal_variance = True

                    takerEnergyFlag, OIdropFlag = takerEnergy.get_taker_energy()
                    # new_average_difference = new_average_difference * modified_sigmoid_smooth(toplongshortratio)
                    # OpenInterestCurrent = await get_open_interest(exchange, futures_symbol)
                    if stopsend:
                        await send_text_message(bot=bot, msg=f"### LongShortRatioDiff: {new_average_difference}, TakerEnergy: {takerEnergyFlag}, OIdrop: {OIdropFlag}, order_book: {open_signal_variance}#{close_signal_R} ###",
                                                chat_id=GP_chat_id)
                        stopsend = False

                    # if (not takerEnergy) and OIdropFlag and Filter_direaction and not open_signal_variance:
                    #     offset, server_time = await sync_server_time(exchange)
                    #     balance = await get_current_balance(exchange, server_time)
                    #
                    #     main_order_side = "BUY"
                    #     main_order_type = "MARKET"
                    #
                    #     position_quantity = 0.004  # Tele_adjust_loss_per_trade*0.003 # max(position_quantity, 0.003)
                    #     position_quantity_list = [0.002, 0.002]
                    #     _, server_time = await sync_server_time(exchange)
                    #     await cancel_ALL_orders(exchange, futures_symbol, server_time)
                    #     main_order_id = await post_futures_order(exchange, futures_symbol, main_order_side, main_order_type,
                    #                                              position_quantity, server_time, PS_chat_id, bot, None)
                    #     OpenInterestCurrent = await get_open_interest(exchange, futures_symbol)
                    #     if OpenInterestCurrent > OpenInterestFirst:
                    #         PositionOIstatus = "UP"
                    #     else:
                    #         PositionOIstatus = "DOWN"
                    #     position_status = await get_position_status(exchange, futures_symbol, server_time)
                    #     if initial_entry_price is None:
                    #         initial_entry_price = float(position_status[0]['entryPrice'])
                    #         grid_size = round(initial_entry_price * 221.87 / 60000, 4)
                    #         grid221_biased = GL.GridLine(initial_entry_price, grid_size, 1000, main_order_side)
                    #         grid221_unbiased = GL.GridLine(initial_entry_price, grid_size, 1000, main_order_side)
                    #
                    #     long_order = {"order_type": "long", "order_price": new_close, "order_quantity": position_quantity,
                    #                   "order_status": "open", "Balance": balance}
                    #     # append the short order to order book text file
                    #     with open('order_book_break.txt', 'a') as f:
                    #         f.write(f"Long Order: {long_order}\n")
                    #     order_book.append(long_order)
                    #     await send_text_message(bot=bot, msg="### opened long position! ###",
                    #                             chat_id=PS_chat_id)
                    #     # TP_order_list = (TP_order_id)
                    #     # SL_order_list = (SL_order_id)
                    #     replaceSL = True
                    #     ReverseOrder = True
                    #     sameKlineOpenPosition = False
                    #     closePosition = False

                    if (not takerEnergyFlag) or OIdropFlag:
                        sameKlineOpenPosition = False
                    # if close_signal_R < 1.1:
                    #     sameKlineOpenPosition = False


                    if new_average_difference >= 0.7 and takerEnergyFlag and not OIdropFlag and open_signal_variance:
                        offset, server_time = await sync_server_time(exchange)
                        balance = await get_current_balance(exchange, server_time)

                        open_order_price_trend_1 = new_close

                        target_SL = max(final_df.iloc[-1]['high'], final_df.iloc[-2]['high'])
                        target_SL = target_SL + target_SL * threshold_percentage / 100
                        # calculate SL ratio
                        SL_ratio = (target_SL - open_order_price_trend_1) / open_order_price_trend_1
                        SL_ratio = abs(SL_ratio)

                        # I adjust SL to lower value to reduce SL loss every trade

                        # target_SL = open_order_price_trend_1 * (1 + SL_moving_ratio)

                        target_TP = open_order_price_trend_1 * (1 - 0.0073)

                        TP_price = round(target_TP)
                        SL_price = round(target_SL)

                        main_order_side = "SELL"
                        TP_order_side = "BUY"
                        SL_order_side = "BUY"
                        main_order_type = "MARKET"
                        TP_order_type = "TAKE_PROFIT_MARKET"
                        SL_order_type = "STOP_MARKET"

                        current_position_size = Tele_adjust_loss_per_trade / SL_ratio

                        if current_position_size > max_position_size:
                            print("margin insufficient")
                            current_position_size = max_position_size

                        position_quantity = round(current_position_size / open_order_price_trend_1, 3)
                        position_quantity = 0.004  # Tele_adjust_loss_per_trade*0.003 # max(position_quantity, 0.003)
                        position_quantity_list = [0.002, 0.002]
                        _, server_time = await sync_server_time(exchange)
                        await cancel_ALL_orders(exchange, futures_symbol, server_time)
                        main_order_id = await post_futures_order(exchange, futures_symbol, main_order_side, main_order_type,
                                                                 position_quantity, server_time, PS_chat_id, bot, None)

                        position_status = await get_position_status(exchange, futures_symbol, server_time)
                        if initial_entry_price is None:
                            initial_entry_price = float(position_status[0]['entryPrice'])
                            grid_size = round(initial_entry_price * 221.87 / 60000, 4)
                            grid221_biased = GL.GridLine(initial_entry_price, grid_size, 1000, main_order_side)
                            grid221_unbiased = GL.GridLine(initial_entry_price, grid_size, 1000, main_order_side)
                        initial_entry_price = None
                        if Filter_direaction:
                            TP_order_id = await post_futures_order(exchange, futures_symbol, TP_order_side, TP_order_type, None,
                                                                server_time, PS_chat_id, bot, TP_price)
                            TP_order_list = (TP_order_id)
                        # while not TP_order_id:
                        #     _, server_time = await sync_server_time(exchange)
                        #     i = 2
                        #     TP_price = round(open_order_price_trend_1 * (1 - (i * SL_ratio * TP_SL_ratio)))
                        #     TP_order_id = await post_futures_order(exchange, futures_symbol, TP_order_side, TP_order_type,
                        #                                            None,
                        #                                            server_time, PS_chat_id, bot, TP_price)
                        #     i += 1

                        # SL_order_id = await post_futures_order(exchange, futures_symbol, SL_order_side, SL_order_type, None,
                        #                                        server_time, PS_chat_id, bot, SL_price)
                        # while not SL_order_id:
                        #     _, server_time = await sync_server_time(exchange)
                        #     i = 2
                        #     SL_price = round(target_SL + i * target_SL * threshold_percentage / 100)
                        #     SL_order_id = await post_futures_order(exchange, futures_symbol, SL_order_side, SL_order_type,
                        #                                            None,
                        #                                            server_time, PS_chat_id, bot, SL_price)
                        #     i += 1

                        short_order = {"order_type": "short", "order_price": new_close, "order_quantity": position_quantity,
                                       "order_status": "open", "Balance": balance}
                        # append the short order to order book text file
                        with open('order_book_break.txt', 'a') as f:
                            f.write(f"Short Order: {short_order}\n")
                        order_book.append(short_order)
                        await send_text_message(bot=bot, msg="### opened short position! ###",
                                                chat_id=PS_chat_id)
                        # TP_order_list = (TP_order_id)
                        # SL_order_list = (SL_order_id)
                        replaceSL = True
                        ReverseOrder = False
                        sameKlineOpenPosition = False
                        closePosition = False
        if res[2]:
            if res[2] != lastCloseLong:
                await send_text_message(bot=bot, msg=f"### Close long signal for {timeframe_merge} period! ###",
                                        chat_id=GP_chat_id)
            if Tele_auto_close_long_order:
                if order_book:
                    if order_book[0]['order_type'] == "long":
                        await close_all_position_at_order_book(exchange, order_book, new_close, balance,
                                                               futures_symbol, bot, PS_chat_id)
                        order_book = []
                        _, server_time = await sync_server_time(exchange)
                        await cancel_ALL_orders(exchange, futures_symbol, server_time)
                        TP_order_list = None
                        SL_order_list = None
        if res[3]:
            if res[3] != lastCloseShort:
                await send_text_message(bot=bot, msg=f"### Close short signal for {timeframe_merge} period! ###",
                                        chat_id=GP_chat_id)
            if Tele_auto_close_short_order:
                if order_book:
                    if order_book[0]['order_type'] == "short":
                        await close_all_position_at_order_book(exchange, order_book, new_close, balance,
                                                               futures_symbol, bot, PS_chat_id)
                        order_book = []
                        _, server_time = await sync_server_time(exchange)
                        await cancel_ALL_orders(exchange, futures_symbol, server_time)
                        TP_order_list = None
                        SL_order_list = None

        # await asyncio.sleep(0.1)
        lastOpenLong = res[0]
        lastOpenShort = res[1]
        lastCloseLong = res[2]
        lastCloseShort = res[3]


async def main():
    # Initialize CCXT for Binance Futures
    # exchange = ccxt.binance({
    #     'apiKey': 'Aa4pqODVec0cA8cXPMyhywCUvALtxRpFGpCCgDmHLyftyZ2hQItWIUZV4EeZAS05',
    #     'secret': 'lqUrBKQiYM6a91LLfwE1IbHwPGqbKX0KPR2y6rA5p2ukrzqltKkU6FxW1on54Y7g',
    #     'enableRateLimit': True  # Recommended by ccxt when making frequent requests
    # })
    exchange = ccxt.binance({
        'apiKey': '',
        'secret': '',
        'enableRateLimit': True  # Recommended by ccxt when making frequent requests
    })

    # Define the trading pair and contract type
    futures_symbol = "BTCUSDT"
    contract_type = 'PERPETUAL'
    leverage = 100
    threshold_percentage = 0.08
    TP_SL_ratio = 8.88
    loss_per_trade = 3
    shortlen = 15
    longlen = 45
    timeframe = '15m'
    use_Volume_filter = False
    max_position_quantity_grid = 0.03
    griz_size_percentage = 0.0
    update_queue = asyncio.Queue()
    command_queue = asyncio.Queue()
    updater = Updater(bot=bot, update_queue=update_queue)
    async with updater:
        updater_task = asyncio.create_task(updater.start_polling())
        processing_task = asyncio.create_task(process_updates(update_queue, bot, command_queue, PS_chat_id))
        trading_task = asyncio.create_task(
            backtest_strategy(exchange, leverage=leverage,
                              futures_symbol=futures_symbol,
                              contract_type=contract_type,
                              threshold_percentage=threshold_percentage, TP_SL_ratio=TP_SL_ratio,
                              loss_per_trade=loss_per_trade,
                              command_queue=command_queue,
                              timeframe_merge=timeframe,
                              use_Volume_filter=use_Volume_filter,
                              shortlen=shortlen,
                              longlen=longlen,
                              max_position_quantity_grid=max_position_quantity_grid
                              ))

        try:
            await asyncio.gather(updater_task, processing_task, trading_task)
        except asyncio.CancelledError:
            print("Tasks were cancelled.")
        finally:
            # Ensure this code runs after the tasks are cancelled
            print("Executing code after the tasks...")

    # asyncio.run(backtest_strategy(exchange, futures_symbol=futures_symbol, contract_type=contract_type, bot=bot, chat_id=chat_id, timeframe_merge=timeframe))


def handle_signal():
    loop.stop()


if __name__ == '__main__':
    # loop = asyncio.get_event_loop()
    # for sig in (signal.SIGINT, signal.SIGTERM):
    #     loop.add_signal_handler(sig, loop.stop)
    # try:
    #     loop.run_until_complete(main())
    # finally:
    #     print("Shutting down...")
    loop = asyncio.get_event_loop()
    try:
        if sys.platform == 'win32':
            loop.add_signal_handler(signal.SIGINT, handle_signal)
        else:
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, handle_signal)
        loop.run_until_complete(main())
    except NotImplementedError:
        if sys.platform == 'win32':
            for sig in (signal.SIGINT, signal.SIGBREAK):
                signal.signal(sig, handle_signal)
        loop.run_until_complete(main())
    finally:
        print("Shutting down...")
        loop.close()
