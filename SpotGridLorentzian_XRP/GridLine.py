import numpy as np
import pandas as pd

class GridLine:
    def __init__(self, initial_price, grid_size, num_levels):
        # Adjust grid initialization to include initial price as both buy and sell level
        # if position_side == "BUY":
        #     buy_levels = [initial_price - i * grid_size for i in range(1, num_levels + 1)]
        #     sell_levels = [initial_price + i * grid_size for i in range(1, num_levels + 1)]
        #     sell_levels.append(initial_price)
        # else:
        #     buy_levels = [initial_price - i * grid_size for i in range(1, num_levels + 1)]
        #     sell_levels = [initial_price + i * grid_size for i in range(1, num_levels + 1)]
        #     buy_levels.append(initial_price)
        buy_levels = [initial_price - i * grid_size for i in range(1, num_levels + 1)]
        sell_levels = [initial_price + i * grid_size for i in range(1, num_levels + 1)]
        buy_levels.append(initial_price)
        grid_levels = sorted(buy_levels + sell_levels)
        self.band = np.array(grid_levels)
        self.gridlabel = [i for i in range(1, 2 * num_levels + 1)]
        self.center = initial_price
        self.last_grid = pd.cut([initial_price], self.band, labels=self.gridlabel, right=False)[0]
        self.first_check = True
        self.first_check_2 = True
        self.grid_change_last = [0, 0]
        self.grid_change_new = [0, 0]
        self.volume = 1
        self.position_long = 0
        self.position_short = 0

    def get_trading_signal(self, price):
        grid = pd.cut([price], self.band, labels=self.gridlabel)[0]
        if self.last_grid < grid:
            if self.first_check:
                if grid != 1000 and grid != 1001:
                    self.grid_change_new = sorted([self.last_grid, grid])
                    self.last_grid = grid

                    self.first_check_2 = False
                    self.first_check = False

                    if self.grid_change_new[0] != self.grid_change_last[0] and self.grid_change_new[1] != \
                            self.grid_change_last[1]:
                        self.grid_change_last = self.grid_change_new

                        return 'sell'
                self.last_grid = grid  # Update first check on the first run
                return 'hold'
            self.grid_change_new = sorted([self.last_grid, grid])
            self.last_grid = grid

            if self.grid_change_new[0] != self.grid_change_last[0] and self.grid_change_new[1] != self.grid_change_last[
                1]:
                self.grid_change_last = self.grid_change_new

                return 'sell'

        elif self.last_grid > grid:
            if self.first_check_2:
                if grid != 1000 and grid != 1001:
                    self.grid_change_new = sorted([self.last_grid, grid])
                    self.last_grid = grid

                    self.first_check_2 = False
                    self.first_check = False

                    if self.grid_change_new[0] != self.grid_change_last[0] and self.grid_change_new[1] != \
                            self.grid_change_last[1]:
                        self.grid_change_last = self.grid_change_new
                        return 'buy'
                self.last_grid = grid  # Update first check on the first run
                return 'hold'
            self.grid_change_new = sorted([self.last_grid, grid])
            self.last_grid = grid

            if self.grid_change_new[0] != self.grid_change_last[0] and self.grid_change_new[1] != self.grid_change_last[
                1]:
                self.grid_change_last = self.grid_change_new
                return 'buy'

        return 'hold'

    def get_trading_signal_long(self, price):
        grid = pd.cut([price], self.band, labels=self.gridlabel)[0]
        if self.last_grid < grid:
            if self.first_check:
                if grid != 1000 and grid != 1001:
                    self.grid_change_new = sorted([self.last_grid, grid])
                    self.last_grid = grid

                    self.first_check_2 = False
                    self.first_check = False

                    if self.grid_change_new[0] != self.grid_change_last[0] and self.grid_change_new[1] != \
                            self.grid_change_last[1]:
                        self.grid_change_last = self.grid_change_new

                        return 'sell'
                self.last_grid = grid  # Update first check on the first run
                return 'hold'
            self.grid_change_new = sorted([self.last_grid, grid])
            self.last_grid = grid

            if self.grid_change_new[0] != self.grid_change_last[0] and self.grid_change_new[1] != self.grid_change_last[
                1]:
                self.grid_change_last = self.grid_change_new

            return 'sell'

        elif self.last_grid > grid:
            if self.first_check_2:
                if grid != 1000 and grid != 1001:
                    self.grid_change_new = sorted([self.last_grid, grid])
                    self.last_grid = grid

                    self.first_check_2 = False
                    self.first_check = False

                    if self.grid_change_new[0] != self.grid_change_last[0] and self.grid_change_new[1] != \
                            self.grid_change_last[1]:
                        self.grid_change_last = self.grid_change_new
                        return 'buy'
                self.last_grid = grid  # Update first check on the first run
                return 'hold'
            self.grid_change_new = sorted([self.last_grid, grid])
            self.last_grid = grid

            if self.grid_change_new[0] != self.grid_change_last[0] and self.grid_change_new[1] != self.grid_change_last[
                1]:
                self.grid_change_last = self.grid_change_new
                return 'buy'

        return 'hold'

    def get_trading_signal_short(self, price):
        grid = pd.cut([price], self.band, labels=self.gridlabel)[0]
        if self.last_grid < grid:
            if self.first_check:
                if grid != 1000 and grid != 1001:
                    self.grid_change_new = sorted([self.last_grid, grid])
                    self.last_grid = grid

                    self.first_check_2 = False
                    self.first_check = False

                    if self.grid_change_new[0] != self.grid_change_last[0] and self.grid_change_new[1] != \
                            self.grid_change_last[1]:
                        self.grid_change_last = self.grid_change_new

                        return 'sell'
                self.last_grid = grid  # Update first check on the first run
                return 'hold'
            self.grid_change_new = sorted([self.last_grid, grid])
            self.last_grid = grid

            if self.grid_change_new[0] != self.grid_change_last[0] and self.grid_change_new[1] != self.grid_change_last[
                1]:
                self.grid_change_last = self.grid_change_new

                return 'sell'

        elif self.last_grid > grid:
            if self.first_check_2:
                if grid != 1000 and grid != 1001:
                    self.grid_change_new = sorted([self.last_grid, grid])
                    self.last_grid = grid

                    self.first_check_2 = False
                    self.first_check = False

                    if self.grid_change_new[0] != self.grid_change_last[0] and self.grid_change_new[1] != \
                            self.grid_change_last[1]:
                        self.grid_change_last = self.grid_change_new
                        return 'buy'
                self.last_grid = grid  # Update first check on the first run
                return 'hold'
            self.grid_change_new = sorted([self.last_grid, grid])
            self.last_grid = grid

            if self.grid_change_new[0] != self.grid_change_last[0] and self.grid_change_new[1] != self.grid_change_last[
                1]:
                self.grid_change_last = self.grid_change_new
            return 'buy'

        return 'hold'