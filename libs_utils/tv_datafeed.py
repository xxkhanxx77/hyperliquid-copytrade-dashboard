import datetime
import enum
import json
import logging
import random
import re
import string
import pandas as pd
from websocket import create_connection
import requests
import json
import time

logger = logging.getLogger(__name__)


class Interval(enum.Enum):
    in_1_minute = "1"
    in_3_minute = "3"
    in_5_minute = "5"
    in_15_minute = "15"
    in_30_minute = "30"
    in_45_minute = "45"
    in_1_hour = "1H"
    in_2_hour = "2H"
    in_3_hour = "3H"
    in_4_hour = "4H"
    in_daily = "1D"
    in_weekly = "1W"
    in_monthly = "1M"


class TvDatafeed:
    __sign_in_url = "https://www.tradingview.com/accounts/signin/"
    __search_url = "https://symbol-search.tradingview.com/symbol_search/?text={}&hl=1&exchange={}&lang=en&type=&domain=production"
    __ws_headers = json.dumps({"Origin": "https://data.tradingview.com"})
    __signin_headers = {"Referer": "https://www.tradingview.com"}
    __ws_timeout = 5

    def __init__(
        self,
        username: str = None,
        password: str = None,
    ) -> None:
        """Create TvDatafeed object
        Args:
            username (str, optional): tradingview username. Defaults to None.
            password (str, optional): tradingview password. Defaults to None.
        """

        self.ws_debug = False

        self.token = self.__auth(username, password)

        if self.token is None:
            self.token = "unauthorized_user_token"
            logger.warning(
                "you are using nologin method, data you access may be limited"
            )

        self.ws = None
        self.session = self.__generate_session()
        self.chart_session = self.__generate_chart_session()

    def __auth(self, username, password):
        if username is None or password is None:
            token = None

        else:
            data = {"username": username, "password": password, "remember": "on"}
            try:
                response = requests.post(
                    url=self.__sign_in_url, data=data, headers=self.__signin_headers
                )
                token = response.json()["user"]["auth_token"]
            except Exception as e:
                logger.error("error while signin")
                token = None

        return token

    def __create_connection(self):
        logging.debug("creating websocket connection")
        self.ws = create_connection(
            "wss://data.tradingview.com/socket.io/websocket",
            headers=self.__ws_headers,
            timeout=self.__ws_timeout,
        )

    @staticmethod
    def __filter_raw_message(text):
        try:
            found = re.search('"m":"(.+?)",', text).group(1)
            found2 = re.search('"p":(.+?"}"])}', text).group(1)

            return found, found2
        except AttributeError:
            logger.error("error in filter_raw_message")

    @staticmethod
    def __generate_session():
        stringLength = 12
        letters = string.ascii_lowercase
        random_string = "".join(random.choice(letters) for i in range(stringLength))
        return "qs_" + random_string

    @staticmethod
    def __generate_chart_session():
        stringLength = 12
        letters = string.ascii_lowercase
        random_string = "".join(random.choice(letters) for i in range(stringLength))
        return "cs_" + random_string

    @staticmethod
    def __prepend_header(st):
        return "~m~" + str(len(st)) + "~m~" + st

    @staticmethod
    def __construct_message(func, param_list):
        return json.dumps({"m": func, "p": param_list}, separators=(",", ":"))

    def __create_message(self, func, paramList):
        return self.__prepend_header(self.__construct_message(func, paramList))

    def __send_message(self, func, args):
        m = self.__create_message(func, args)
        if self.ws_debug:
            print(m)
        self.ws.send(m)

    @staticmethod
    def __create_df(raw_data, symbol):
        try:
            out = re.search('"s":\[(.+?)\}\]', raw_data).group(1)
            x = out.split(',{"')
            data = list()
            volume_data = True

            for xi in x:
                xi = re.split("\[|:|,|\]", xi)
                ts = datetime.datetime.fromtimestamp(float(xi[4]))

                row = [ts]

                for i in range(5, 10):
                    # skip converting volume data if does not exists
                    if not volume_data and i == 9:
                        row.append(0.0)
                        continue
                    try:
                        row.append(float(xi[i]))

                    except ValueError:
                        volume_data = False
                        row.append(0.0)
                        logger.debug("no volume data")

                data.append(row)

            data = pd.DataFrame(
                data, columns=["datetime", "open", "high", "low", "close", "volume"]
            ).set_index("datetime")
            data.insert(0, "symbol", value=symbol)
            return data
        except AttributeError:
            logger.error("no data, please check the exchange and symbol")

    @staticmethod
    def __format_symbol(symbol, exchange, contract: int = None):
        if ":" in symbol:
            pass
        elif contract is None:
            symbol = f"{exchange}:{symbol}"

        elif isinstance(contract, int):
            symbol = f"{exchange}:{symbol}{contract}!"

        else:
            raise ValueError("not a valid contract")

        return symbol

    def get_hist(
        self,
        symbol: str,
        exchange: str = "NSE",
        interval: Interval = Interval.in_daily,
        n_bars: int = 10,
        fut_contract: int = None,
        extended_session: bool = False,
    ) -> pd.DataFrame:
        """get historical data
        Args:
            symbol (str): symbol name
            exchange (str, optional): exchange, not required if symbol is in format EXCHANGE:SYMBOL. Defaults to None.
            interval (str, optional): chart interval. Defaults to 'D'.
            n_bars (int, optional): no of bars to download, max 5000. Defaults to 10.
            fut_contract (int, optional): None for cash, 1 for continuous current contract in front, 2 for continuous next contract in front . Defaults to None.
            extended_session (bool, optional): regular session if False, extended session if True, Defaults to False.
        Returns:
            pd.Dataframe: dataframe with sohlcv as columns
        """
        symbol = self.__format_symbol(
            symbol=symbol, exchange=exchange, contract=fut_contract
        )

        interval = interval.value

        self.__create_connection()

        self.__send_message("set_auth_token", [self.token])
        self.__send_message("chart_create_session", [self.chart_session, ""])
        self.__send_message("quote_create_session", [self.session])
        self.__send_message(
            "quote_set_fields",
            [
                self.session,
                "ch",
                "chp",
                "current_session",
                "description",
                "local_description",
                "language",
                "exchange",
                "fractional",
                "is_tradable",
                "lp",
                "lp_time",
                "minmov",
                "minmove2",
                "original_name",
                "pricescale",
                "pro_name",
                "short_name",
                "type",
                "update_mode",
                "volume",
                "currency_code",
                "rchp",
                "rtc",
            ],
        )

        self.__send_message(
            "quote_add_symbols", [self.session, symbol, {"flags": ["force_permission"]}]
        )
        self.__send_message("quote_fast_symbols", [self.session, symbol])

        self.__send_message(
            "resolve_symbol",
            [
                self.chart_session,
                "symbol_1",
                '={"symbol":"'
                + symbol
                + '","adjustment":"splits","session":'
                + ('"regular"' if not extended_session else '"extended"')
                + "}",
            ],
        )
        self.__send_message(
            "create_series",
            [self.chart_session, "s1", "s1", "symbol_1", interval, n_bars],
        )
        self.__send_message("switch_timezone", [self.chart_session, "exchange"])

        raw_data = ""

        logger.debug(f"getting data for {symbol}...")
        while True:
            try:
                result = self.ws.recv()
                raw_data = raw_data + result + "\n"
            except Exception as e:
                logger.error(e)
                break

            if "series_completed" in result:
                break

        return self.__create_df(raw_data, symbol)

    def search_symbol(self, text: str, exchange: str = ""):
        url = self.__search_url.format(text, exchange)

        symbols_list = []
        try:
            resp = requests.get(url)

            symbols_list = json.loads(
                resp.text.replace("</em>", "").replace("<em>", "")
            )
        except Exception as e:
            logger.error(e)

        return symbols_list

    def get_all_crypto_symbols(self, exchange: str = "BINANCE"):
        """Get ALL crypto ticker symbols from specific exchange using TradingView screener
        
        Args:
            exchange (str): Exchange name (BINANCE, COINBASE, KRAKEN, etc.)
            
        Returns:
            list: Complete list of crypto symbols with metadata
        """
        # TradingView screener endpoint for crypto
        screener_url = "https://scanner.tradingview.com/crypto/scan"
        
        symbols = []
        offset = 0
        limit = 100  # Maximum per request
        
        while True:
            payload = {
                "filter": [
                    {
                        "left": "exchange",
                        "operation": "equal", 
                        "right": exchange
                    }
                ],
                "options": {
                    "lang": "en"
                },
                "symbols": {
                    "query": {
                        "types": []
                    },
                    "tickers": []
                },
                "columns": [
                    "name",
                    "description", 
                    "logoid",
                    "update_mode",
                    "type",
                    "typespecs",
                    "exchange",
                    "currency",
                    "fundamental_currency_code"
                ],
                "sort": {
                    "sortBy": "name",
                    "sortOrder": "asc"
                },
                "range": [offset, offset + limit]
            }
            
            try:
                response = requests.post(screener_url, json=payload, timeout=10)
                
                if response.status_code != 200:
                    logger.error(f"Screener API returned status {response.status_code}")
                    break
                    
                data = response.json()
                
                if 'data' not in data or not data['data']:
                    break
                    
                batch_symbols = []
                for item in data['data']:
                    symbol_info = {
                        'symbol': item['s'],  # Full format like BINANCE:BTCUSDT
                        'name': item['d'][0] if item['d'] else '',
                        'description': item['d'][1] if len(item['d']) > 1 else '',
                        'exchange': exchange,
                        'ticker_only': item['s'].split(':')[1] if ':' in item['s'] else item['s']
                    }
                    batch_symbols.append(symbol_info)
                
                symbols.extend(batch_symbols)
                
                # Check if we got fewer results than requested (last page)
                if len(batch_symbols) < limit:
                    break
                    
                offset += limit
                
                # Rate limiting - small delay between requests
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error fetching crypto symbols: {e}")
                break
                
        logger.info(f"Retrieved {len(symbols)} symbols from {exchange}")
        return symbols

    def get_crypto_symbols_from_binance_api(self):
        """Alternative method: Get symbols directly from Binance API and format for TradingView
        
        Returns:
            list: Complete list of crypto symbols formatted for TradingView
        """
        try:
            # Binance public API endpoint
            binance_url = "https://api.binance.com/api/v3/exchangeInfo"
            response = requests.get(binance_url, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"Binance API returned status {response.status_code}")
                return []
            
            data = response.json()
            symbols = []
            
            for symbol_info in data['symbols']:
                if symbol_info['status'] == 'TRADING':
                    tv_symbol = f"BINANCE:{symbol_info['symbol']}"
                    symbol_data = {
                        'symbol': tv_symbol,
                        'binance_symbol': symbol_info['symbol'],
                        'base_asset': symbol_info['baseAsset'],
                        'quote_asset': symbol_info['quoteAsset'],
                        'status': symbol_info['status'],
                        'ticker_only': symbol_info['symbol']
                    }
                    symbols.append(symbol_data)
            
            logger.info(f"Retrieved {len(symbols)} trading symbols from Binance API")
            return symbols
            
        except Exception as e:
            logger.error(f"Error fetching from Binance API: {e}")
            return []

    def get_spot_crypto_symbols(self, exchange: str = "BINANCE"):
        """Get SPOT crypto ticker symbols only
        
        Args:
            exchange (str): Exchange name (BINANCE, COINBASE, KRAKEN, etc.)
            
        Returns:
            list: List of SPOT crypto symbols only
        """
        all_symbols = self.get_all_crypto_symbols(exchange)
        
        # Filter only spot symbols (exclude perpetual futures and other derivatives)
        spot_symbols = []
        for symbol in all_symbols:
            ticker = symbol['ticker_only']
            
            # Skip perpetual futures (.P suffix), futures contracts, and other derivatives
            if (not ticker.endswith('.P') and 
                not ticker.endswith('USD_PERP') and
                not ticker.endswith('USDT_PERP') and
                not any(month in ticker for month in ['0325', '0626', '0927', '1228']) and  # Quarterly futures
                not ticker.endswith('_FUT') and
                not re.search(r'\d{6}$', ticker)):  # YYMMDD format futures
                
                spot_symbols.append(symbol)
        
        logger.info(f"Filtered {len(spot_symbols)} SPOT symbols from {len(all_symbols)} total symbols")
        return spot_symbols

    def get_perpetual_crypto_symbols(self, exchange: str = "BINANCE"):
        """Get PERPETUAL FUTURES crypto ticker symbols only
        
        Args:
            exchange (str): Exchange name (BINANCE, BYBIT, etc.)
            
        Returns:
            list: List of PERPETUAL futures crypto symbols only
        """
        all_symbols = self.get_all_crypto_symbols(exchange)
        
        # Filter only perpetual futures symbols
        perp_symbols = []
        for symbol in all_symbols:
            ticker = symbol['ticker_only']
            
            # Include perpetual futures (.P suffix for Binance, _PERP for others)
            if (ticker.endswith('.P') or 
                ticker.endswith('USD_PERP') or
                ticker.endswith('USDT_PERP')):
                
                perp_symbols.append(symbol)
        
        logger.info(f"Filtered {len(perp_symbols)} PERPETUAL symbols from {len(all_symbols)} total symbols")
        return perp_symbols

    def get_spot_symbols_from_binance_api(self):
        """Get SPOT symbols only from Binance API directly
        
        Returns:
            list: Complete list of SPOT crypto symbols from Binance
        """
        try:
            binance_url = "https://api.binance.com/api/v3/exchangeInfo"
            response = requests.get(binance_url, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"Binance API returned status {response.status_code}")
                return []
            
            data = response.json()
            spot_symbols = []
            
            for symbol_info in data['symbols']:
                # Only include SPOT trading symbols
                if (symbol_info['status'] == 'TRADING' and 
                    symbol_info.get('contractType') is None):  # SPOT has no contractType
                    
                    tv_symbol = f"BINANCE:{symbol_info['symbol']}"
                    symbol_data = {
                        'symbol': tv_symbol,
                        'binance_symbol': symbol_info['symbol'],
                        'base_asset': symbol_info['baseAsset'],
                        'quote_asset': symbol_info['quoteAsset'],
                        'status': symbol_info['status'],
                        'ticker_only': symbol_info['symbol'],
                        'type': 'SPOT'
                    }
                    spot_symbols.append(symbol_data)
            
            logger.info(f"Retrieved {len(spot_symbols)} SPOT symbols from Binance API")
            return spot_symbols
            
        except Exception as e:
            logger.error(f"Error fetching SPOT symbols from Binance API: {e}")
            return []

    def get_perpetual_symbols_from_binance_api(self):
        """Get PERPETUAL FUTURES symbols from Binance Futures API directly
        
        Returns:
            list: Complete list of PERPETUAL futures symbols from Binance
        """
        try:
            # Binance Futures API endpoint
            binance_futures_url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
            response = requests.get(binance_futures_url, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"Binance Futures API returned status {response.status_code}")
                return []
            
            data = response.json()
            perp_symbols = []
            
            for symbol_info in data['symbols']:
                # Only include PERPETUAL contracts that are trading
                if (symbol_info['status'] == 'TRADING' and 
                    symbol_info.get('contractType') == 'PERPETUAL'):
                    
                    tv_symbol = f"BINANCE:{symbol_info['symbol']}.P"  # Add .P for TradingView format
                    symbol_data = {
                        'symbol': tv_symbol,
                        'binance_symbol': symbol_info['symbol'],
                        'base_asset': symbol_info['baseAsset'],
                        'quote_asset': symbol_info['quoteAsset'],
                        'status': symbol_info['status'],
                        'ticker_only': f"{symbol_info['symbol']}.P",
                        'type': 'PERPETUAL',
                        'contract_type': symbol_info.get('contractType'),
                        'underlying_type': symbol_info.get('underlyingType')
                    }
                    perp_symbols.append(symbol_data)
            
            logger.info(f"Retrieved {len(perp_symbols)} PERPETUAL symbols from Binance Futures API")
            return perp_symbols
            
        except Exception as e:
            logger.error(f"Error fetching PERPETUAL symbols from Binance Futures API: {e}")
            return []

    def get_filtered_symbols_by_quote_asset(self, exchange: str = "BINANCE", quote_assets: list = None, symbol_type: str = "SPOT"):
        """Get crypto symbols filtered by quote asset (USDT, USDC, BTC, etc.)
        
        Args:
            exchange (str): Exchange name
            quote_assets (list): List of quote assets to filter (e.g., ['USDT', 'USDC', 'BTC'])
            symbol_type (str): 'SPOT', 'PERPETUAL', or 'ALL'
            
        Returns:
            list: Filtered crypto symbols
        """
        if quote_assets is None:
            quote_assets = ['USDT']  # Default to USDT pairs
        
        # Get symbols based on type
        if symbol_type == "SPOT":
            if exchange == "BINANCE":
                symbols = self.get_spot_symbols_from_binance_api()
            else:
                symbols = self.get_spot_crypto_symbols(exchange)
        elif symbol_type == "PERPETUAL":
            if exchange == "BINANCE":
                symbols = self.get_perpetual_symbols_from_binance_api()
            else:
                symbols = self.get_perpetual_crypto_symbols(exchange)
        else:  # ALL
            symbols = self.get_all_crypto_symbols(exchange)
        
        # Filter by quote assets
        filtered_symbols = []
        for symbol in symbols:
            if hasattr(symbol, 'get') and symbol.get('quote_asset'):
                # Use quote_asset field if available (from Binance API)
                if symbol['quote_asset'] in quote_assets:
                    filtered_symbols.append(symbol)
            else:
                # Fallback to ticker parsing (from TradingView API)
                ticker = symbol.get('ticker_only', '')
                for quote in quote_assets:
                    if ticker.endswith(quote) or ticker.endswith(f"{quote}.P"):
                        filtered_symbols.append(symbol)
                        break
        
        logger.info(f"Filtered {len(filtered_symbols)} symbols with quote assets {quote_assets}")
        return filtered_symbols

    def get_available_crypto_exchanges(self):
        """Get list of available crypto exchanges in TradingView
        
        Returns:
            list: List of exchange names available for crypto trading
        """
        exchanges = [
            "BINANCE",
            "COINBASE", 
            "KRAKEN",
            "BITFINEX",
            "HUOBI",
            "OKEX",
            "KUCOIN",
            "GATEIO",
            "BITGET",
            "MEXC",
            "CRYPTOCOM",
            "GEMINI",
            "BITSTAMP",
            "BYBIT",
            "PHEMEX",
            "DERIBIT"
        ]
        return exchanges


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    tv = TvDatafeed()
    print(tv.get_hist("CRUDEOIL", "MCX", fut_contract=1))
    print(tv.get_hist("NIFTY", "NSE", fut_contract=1))
    print(
        tv.get_hist(
            "EICHERMOT",
            "NSE",
            interval=Interval.in_1_hour,
            n_bars=500,
            extended_session=False,
        )
    )



#   📋 สรุปตาราง Functions และ Data Source:

#   | Function                                 | Exchange     | Data Source           | ใช้ TV? |
#   |------------------------------------------|--------------|-----------------------|---------|
#   | get_spot_symbols_from_binance_api()      | Binance      | Binance API           | ❌       |
#   | get_perpetual_symbols_from_binance_api() | Binance      | Binance Futures API   | ❌       |
#   | get_spot_crypto_symbols()                | อื่นๆ        | TradingView Screener  | ✅       |
#   | get_perpetual_crypto_symbols()           | อื่นๆ        | TradingView Screener  | ✅       |
#   | get_all_crypto_symbols()                 | ทุก exchange | TradingView Screener  | ✅       |
#   | get_filtered_symbols_by_quote_asset()    | Mixed        | Mixed (TV หรือ API)   | ✅/❌     |
#   | search_symbol()                          | ทุก exchange | TradingView Search    | ✅       |
#   | get_hist()                               | ทุก exchange | TradingView WebSocket | ✅       |