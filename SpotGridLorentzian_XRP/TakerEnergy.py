import numpy as np
import threading
import ccxt
import datetime
from time import sleep

class TakerEnergy:
    def __init__(self, exchange, futures_symbol):
        self.exchange = exchange
        self.futures_symbol = futures_symbol
        self.contract_type = 'PERPETUAL'
        self.time_interval = 5 # 5 seconds

        #define a buffer with length of 180
        self.buy_buffer = np.zeros(180)
        self.sell_buffer = np.zeros(180)
        self.open_interest = np.zeros(180)
        self.running = True
        self.thread = threading.Thread(target=self.taker_buy_sell_loop)
        self.thread.daemon = True  # Allows program to exit even if this thread is running
        self.thread.start()


        
    def taker_buy_sell_loop(self):
        First_second_of_interval = None
        offset, server_time = self.sync_server_time(self.exchange)
        LastTime = self.is_first_second_of_minute(server_time, self.time_interval)
        while True:
            trades = self.get_recent_trades(self.exchange, self.futures_symbol)
            latest_time = float(trades[-1]['time'])
            NewTime = self.is_first_second_of_minute(latest_time, self.time_interval)
            if NewTime != LastTime:
                First_second_of_interval = True
            else:
                First_second_of_interval = False
            #LastTime = NewTime
            if First_second_of_interval:

                #Taker long short value
                buy_volume = 0
                sell_volume = 0
                #loop trades reversely
                for trade in trades[::-1]:
                    trade_time = float(trade['time'])
                    if self.is_first_second_of_minute(trade_time, self.time_interval) == LastTime:
                        if trade['isBuyerMaker']:
                            sell_volume += float(trade['qty'])
                        else:
                            buy_volume += float(trade['qty'])
                price = float(trades[-1]['price'])
                
                sell_volume = round(sell_volume, 3)
                buy_volume = round(buy_volume, 3)
                
                #put the buy and sell volume into the buffer and make sure the buffer length is 180
                self.buy_buffer = np.roll(self.buy_buffer, 1)
                self.buy_buffer[0] = buy_volume
                self.sell_buffer = np.roll(self.sell_buffer, 1)
                self.sell_buffer[0] = sell_volume

                open_interest, OItime = self.get_open_interest(self.exchange, self.futures_symbol)
                self.open_interest = np.roll(self.open_interest, 1)
                self.open_interest[0] = open_interest

            LastTime = NewTime
            sleep(1)

    @staticmethod
    def get_recent_trades(exchange, futures_symbol):
        while True:
            try:
                trades = exchange.fapiPublicGetTrades({"symbol": futures_symbol, "limit": 1000})
                return trades
            except Exception as e:
                print(f"Error fetching trades from server: {e}")
                continue
        return
    
    @staticmethod
    def get_open_interest(exchange, futures_symbol):
        while True:
            try:
                status = exchange.fetchOpenInterest(futures_symbol)
                return float(status['openInterestAmount']), float(status['timestamp'])
            except Exception as e:
                print(f"get status: {e}")
                continue

    @staticmethod
    def is_first_second_of_minute(kline_server_time, minutes):
        # Convert server time from milliseconds to seconds
        server_time_seconds = kline_server_time / 1000.0 / (minutes)
        # Convert to a datetime object
        # dt = datetime.datetime.fromtimestamp(server_time_seconds)
        # Check if the minute is zero
        return int(server_time_seconds)

    @staticmethod
    def sync_server_time(exchange):
        while True:
            try:
                server_time = exchange.fapiPublicGetTime()
                server_time = int(server_time['serverTime'])
                local_time = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
                return server_time - local_time, server_time
            except Exception as e:
                print(f"Error fetching time from server: {e}")
                continue
        return

    def get_taker_energy(self):
        #normalize the buy_buffer by deviding 500
        norm_buy = self.buy_buffer / 500
        energy_buy = np.sum(norm_buy**2)
        #normalize the sell_buffer by deviding 500
        norm_sell = self.sell_buffer / 500
        energy_sell = np.sum(norm_sell**2)

        # return energy_buy, energy_sell
        OIdrop = False

        norm_open_interest = (self.open_interest - self.open_interest.min()) / (self.open_interest.max() - self.open_interest.min())
        #using a sliding window with length of 10 to calculate the difference bewteen the max and min in the window
        diff_open_interest = np.zeros(161)
        for i in range(161):
            diff_open_interest[i] = norm_open_interest[i:i+20].max() - norm_open_interest[i:i+20].min()
            min_index = np.argmin(norm_open_interest[i:i+20])
            max_index = np.argmax(norm_open_interest[i:i+20])
            if diff_open_interest[i] > 0.65 and max_index > min_index and self.open_interest.max()-self.open_interest.min() > 270:
                OIdrop = True
                break

        takerEnergyFlag = False
        if energy_buy > 0.07 and energy_sell > 0.07:
            takerEnergyFlag = True


        return takerEnergyFlag, OIdrop
