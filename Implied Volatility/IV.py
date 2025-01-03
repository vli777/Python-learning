#SVI Implied Volatility Surface

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import newton, minimize
from datetime import datetime as dt
import re
import matplotlib.pyplot as plt

class OptionsPricing:
    """
    A class to handle options pricing, implied volatility calculations,
    and SVI (Stochastic Volatility Inspired) surface fitting.
    """

    def __init__(self, spot_price: float, risk_free_rate: float, dividend_yield: float):
        self.spot_price = spot_price
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield

    @staticmethod
    def black_scholes(S: float, K: float, T: float, v: float, r: float, q: float, option_type: str) -> float:
        """
        Calculate the Black-Scholes price for an option.
        """
        d1 = (np.log(S / K) + (r - q + 0.5 * v ** 2) * T) / (v * np.sqrt(T))
        d2 = d1 - v * np.sqrt(T)

        if option_type == "call":
            return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == "put":
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")

    def implied_volatility(self, S: float, K: float, T: float, market_price: float, option_type: str) -> float:
        """
        Calculate implied volatility using Newton's method.
        """
        initial_vol = 0.2

        def objective_function(vol):
            return self.black_scholes(S, K, T, vol, self.risk_free_rate, self.dividend_yield, option_type) - market_price

        return newton(objective_function, initial_vol)

    @staticmethod
    def calculate_rates(S: float, K: np.ndarray, T: float, call_prices: np.ndarray, put_prices: np.ndarray) -> tuple:
        """
        Calculate risk-free rate and dividend yield.
        """
        m, b = np.polyfit(K, call_prices - put_prices, 1)
        r = -np.log(-m) / T
        q = -np.log(b / S) / T
        return r, q

    @staticmethod
    def svi(a: float, b: float, p: float, m: float, s: float, K: float, S: float) -> float:
        """
        Stochastic Volatility Inspired (SVI) parameterized volatility function.
        """
        x = np.log(K / S)
        return a + b * (p * (x - m) + np.sqrt((x - m) ** 2 + s ** 2))

    @staticmethod
    def svi_fit(params: np.ndarray, strikes: np.ndarray, ivs: np.ndarray, S: float) -> float:
        """
        Objective function to fit SVI parameters.
        """
        svi_variances = np.array([OptionsPricing.svi(*params, K, S) for K in strikes])
        return np.sum((svi_variances - ivs ** 2) ** 2)

    def fit_svi(self, strikes: np.ndarray, ivs: np.ndarray) -> np.ndarray:
        """
        Fit the SVI parameters to the data.
        """
        initial_params = [0.01, 0.1, 0.1, 0.1, 0.1]
        result = minimize(self.svi_fit, initial_params, args=(strikes, ivs, self.spot_price))
        return result.x

    @staticmethod
    def load_option_chain(file_path: str) -> tuple:
        """
        Load and process the option chain data from a file.
        """
        with open(file_path, 'r') as file:
            spot_price = float(file.readline().split(',')[1])
            current_date = pd.to_datetime(file.readline().strip()[:11])

            df = pd.read_csv(file_path, skiprows=2).iloc[:, :-1]
            df.rename(columns={"Bid": "Cbid", "Ask": "Cask", "Bid.1": "Pbid", "Ask.1": "Pask"}, inplace=True)

            maturity_row = df['Calls'].iloc[0]
            maturity_date = OptionsPricing.parse_maturity(maturity_row)
            time_to_maturity = (maturity_date - current_date).days / 365

            df['K'] = df['Calls'].apply(lambda x: float(x.split(' ')[2]))
            df['C'] = (df['Cbid'] + df['Cask']) / 2
            df['P'] = (df['Pbid'] + df['Pask']) / 2

            return spot_price, time_to_maturity, df

    @staticmethod
    def parse_maturity(option_string: str) -> dt:
        """
        Parse the maturity date from the option string.
        """
        pattern = re.compile(r"(\d{2})(\w{3})(\d{2})")
        match = pattern.search(option_string)

        if not match:
            raise ValueError("Invalid option string format.")

        day, month, year = match.groups()
        year = str(int(year) + 2000)
        return dt.strptime(f"{month} {day} {year} 16:00", '%b %d %Y %H:%M')

# Main Execution
if __name__ == "__main__":
    spot_price, T, chain = OptionsPricing.load_option_chain('./quotedata.dat')

    pricing = OptionsPricing(spot_price, 0.0, 0.0)
    r, q = OptionsPricing.calculate_rates(spot_price, chain['K'], T, chain['C'], chain['P'])

    chain['cvol'] = chain.apply(lambda row: pricing.implied_volatility(spot_price, row['K'], T, row['C'], 'call'), axis=1)
    chain['pvol'] = chain.apply(lambda row: pricing.implied_volatility(spot_price, row['K'], T, row['P'], 'put'), axis=1)
    chain['IV'] = (chain['cvol'] + chain['pvol']) / 2

    params = pricing.fit_svi(chain['K'], chain['IV'])

    fit_ivs = np.sqrt([OptionsPricing.svi(*params, K, spot_price) for K in chain['K']])

    mae = np.mean(np.abs(chain['IV'] - fit_ivs))
    rmse = np.sqrt(np.mean((chain['IV'] - fit_ivs) ** 2))

    print(f"Risk-Free Rate: {r:.4f}, Dividend Yield: {q:.4f}")
    print(f"Fitted SVI Parameters: {params}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    plt.plot(chain['K'], chain['IV'], label="Market IV", linestyle='--')
    plt.plot(chain['K'], fit_ivs, label="Fitted IV", linestyle='-')
    plt.xlabel("Strike Price")
    plt.ylabel("Implied Volatility")
    plt.legend()
    plt.title("Implied Volatility Surface Fit")
    plt.show()
