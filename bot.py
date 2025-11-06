import os
import requests
import json
import time
import re
import pytz
from datetime import datetime
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

# Colorama setup
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    print("Warning: Colorama not installed. Run: pip install colorama")

# Load environment variables
load_dotenv()

# Global color variables for fallback
if not COLORAMA_AVAILABLE:
    class DummyColors:
        def __getattr__(self, name):
            return ''
    Fore = DummyColors()
    Back = DummyColors() 
    Style = DummyColors()

class AggressiveThreeMinScalpingBot:
    def __init__(self):
        # Load config from .env file
        self.binance_api_key = os.getenv('BINANCE_API_KEY')
        self.binance_secret = os.getenv('BINANCE_SECRET_KEY')
        self.openrouter_key = os.getenv('OPENROUTER_API_KEY')  # OpenRouter key
        
        # Store colorama references
        self.Fore = Fore
        self.Back = Back
        self.Style = Style
        self.COLORAMA_AVAILABLE = COLORAMA_AVAILABLE
        
        # Thailand timezone
        self.thailand_tz = pytz.timezone('Asia/Bangkok')
        
        # FIXED SETTINGS AS REQUESTED
        self.trade_size_usd = 50
        self.default_leverage = 5
        self.max_concurrent_trades = 8
        self.available_pairs = ["SOLUSDT", "XRPUSDT", "LINKUSDT", "DOGEUSDT", "SUIUSDT"]
        
        # Track bot-opened trades only
        self.bot_opened_trades = {}
        
        # Trade history
        self.trade_history_file = "aggressive_3min_scalping_history.json"
        self.trade_history = self.load_trade_history()
        
        # Precision settings
        self.quantity_precision = {}
        self.price_precision = {}
        
        # Trading statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        
        # Initialize Binance client
        try:
            self.binance = Client(self.binance_api_key, self.binance_secret)
            self.print_color("FULL AI CONTROL 3MIN SCALPING BOT ACTIVATED!", self.Fore.RED + self.Style.BRIGHT)
            self.print_color("AI DECIDES: Entry, TP, SL | FIXED: $50 | 5x | SOL XRP LINK DOGE SUI", self.Fore.CYAN + self.Style.BRIGHT)
        except Exception as e:
            self.print_color(f"Binance initialization failed: {e}", self.Fore.RED)
            self.binance = None
        
        self.validate_config()
        if self.binance:
            self.setup_futures()
            self.load_symbol_precision()
    
    def load_trade_history(self):
        try:
            if os.path.exists(self.trade_history_file):
                with open(self.trade_history_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            self.print_color(f"Error loading trade history: {e}", self.Fore.RED)
            return []
    
    def save_trade_history(self):
        try:
            with open(self.trade_history_file, 'w') as f:
                json.dump(self.trade_history, f, indent=2)
        except Exception as e:
            self.print_color(f"Error saving trade history: {e}", self.Fore.RED)
    
    def add_trade_to_history(self, trade_data):
        try:
            trade_data['close_time'] = self.get_thailand_time()
            trade_data['close_timestamp'] = time.time()
            self.trade_history.append(trade_data)
            
            self.total_trades += 1
            pnl = trade_data.get('pnl', 0)
            self.total_pnl += pnl
            if pnl > 0:
                self.winning_trades += 1
                
            if len(self.trade_history) > 200:
                self.trade_history = self.trade_history[-200:]
            self.save_trade_history()
            self.print_color(f"Trade saved: {trade_data['pair']} {trade_data['direction']} P&L: ${pnl:.2f}", self.Fore.CYAN)
        except Exception as e:
            self.print_color(f"Error adding trade to history: {e}", self.Fore.RED)
    
    def show_trade_history(self, limit=15):
        if not self.trade_history:
            self.print_color("No trade history found", self.Fore.YELLOW)
            return
        
        self.print_color(f"\nFULL AI TRADING HISTORY (Last {min(limit, len(self.trade_history))} trades)", self.Fore.RED + self.Style.BRIGHT)
        self.print_color("=" * 100, self.Fore.RED)
        
        recent_trades = self.trade_history[-limit:]
        for i, trade in enumerate(reversed(recent_trades)):
            pnl = trade.get('pnl', 0)
            pnl_color = self.Fore.GREEN + self.Style.BRIGHT if pnl > 0 else self.Fore.RED + self.Style.BRIGHT if pnl < 0 else self.Fore.YELLOW
            direction_icon = "LONG" if trade['direction'] == 'LONG' else "SHORT"
            close_reason = trade.get('close_reason', 'AI')
            
            self.print_color(f"{i+1:2d}. {direction_icon} {trade['pair']} | Entry: ${trade.get('entry_price', 0):.4f} | Exit: ${trade.get('exit_price', 0):.4f} | P&L: ${pnl:.2f}", pnl_color)
            self.print_color(f"     TP: ${trade.get('take_profit', 0):.4f} | SL: ${trade.get('stop_loss', 0):.4f} | Qty: {trade.get('quantity', 0)} | {close_reason}", self.Fore.YELLOW)
    
    def show_trading_stats(self):
        if self.total_trades == 0:
            return
            
        win_rate = (self.winning_trades / self.total_trades) * 100
        avg_trade = self.total_pnl / self.total_trades
        
        self.print_color(f"\nLIVE TRADING STATISTICS", self.Fore.CYAN + self.Style.BRIGHT)
        self.print_color("=" * 60, self.Fore.CYAN)
        self.print_color(f"Total Trades: {self.total_trades} | Winning Trades: {self.winning_trades}", self.Fore.WHITE)
        self.print_color(f"Win Rate: {win_rate:.1f}%", self.Fore.GREEN + self.Style.BRIGHT if win_rate > 50 else self.Fore.YELLOW)
        self.print_color(f"Total P&L: ${self.total_pnl:.2f}", self.Fore.GREEN + self.Style.BRIGHT if self.total_pnl > 0 else self.Fore.RED + self.Style.BRIGHT)
        self.print_color(f"Average P&L per Trade: ${avg_trade:.2f}", self.Fore.WHITE)
    
    def get_thailand_time(self):
        now_utc = datetime.now(pytz.utc)
        thailand_time = now_utc.astimezone(self.thailand_tz)
        return thailand_time.strftime('%Y-%m-%d %H:%M:%S')
    
    def print_color(self, text, color="", style=""):
        if self.COLORAMA_AVAILABLE:
            print(f"{style}{color}{text}")
        else:
            print(text)
    
    def validate_config(self):
        if not all([self.binance_api_key, self.binance_secret, self.openrouter_key]):
            self.print_color("Missing API keys!", self.Fore.RED)
            return False
        try:
            if self.binance:
                self.binance.futures_exchange_info()
                self.print_color("Binance connection successful!", self.Fore.GREEN + self.Style.BRIGHT)
            else:
                self.print_color("Binance client not available - Paper Trading only", self.Fore.YELLOW)
                return True
        except Exception as e:
            self.print_color(f"Binance connection failed: {e}", self.Fore.RED)
            return False
        return True

    def setup_futures(self):
        if not self.binance:
            return
        try:
            for pair in self.available_pairs:
                try:
                    self.binance.futures_change_leverage(symbol=pair, leverage=self.default_leverage)
                    self.binance.futures_change_margin_type(symbol=pair, marginType='ISOLATED')
                    self.print_color(f"Default 5x leverage set for {pair}", self.Fore.GREEN)
                except Exception as e:
                    self.print_color(f"Leverage setup failed for {pair}: {e}", self.Fore.YELLOW)
            self.print_color("Futures setup completed!", self.Fore.GREEN + self.Style.BRIGHT)
        except Exception as e:
            self.print_color(f"Futures setup failed: {e}", self.Fore.RED)
    
    def load_symbol_precision(self):
        if not self.binance:
            for pair in self.available_pairs:
                self.quantity_precision[pair] = 3
                self.price_precision[pair] = 4
            self.print_color("Default precision set for paper trading", self.Fore.GREEN)
            return
        try:
            exchange_info = self.binance.futures_exchange_info()
            for symbol in exchange_info['symbols']:
                pair = symbol['symbol']
                if pair not in self.available_pairs:
                    continue
                for f in symbol['filters']:
                    if f['filterType'] == 'LOT_SIZE':
                        step_size = f['stepSize']
                        qty_precision = len(step_size.split('.')[1].rstrip('0')) if '.' in step_size else 0
                        self.quantity_precision[pair] = qty_precision
                    elif f['filterType'] == 'PRICE_FILTER':
                        tick_size = f['tickSize']
                        price_precision = len(tick_size.split('.')[1].rstrip('0')) if '.' in tick_size else 0
                        self.price_precision[pair] = price_precision
            self.print_color("Symbol precision loaded", self.Fore.GREEN + self.Style.BRIGHT)
        except Exception as e:
            self.print_color(f"Error loading symbol precision: {e}", self.Fore.RED)
    
    def format_price(self, pair, price):
        if price <= 0:
            return 0.0
        precision = self.price_precision.get(pair, 4)
        return round(price, precision)
    
    def format_quantity(self, pair, quantity):
        if quantity <= 0:
            return 0.0
        precision = self.quantity_precision.get(pair, 3)
        return round(quantity, precision)

    def parse_ai_response(self, text):
        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                decision_data = json.loads(json_str)
                
                direction = decision_data.get('direction', 'HOLD').upper()
                if direction not in ['LONG', 'SHORT', 'HOLD']:
                    direction = 'HOLD'
                
                entry_price = float(decision_data.get('entry_price', 0))
                if entry_price <= 0:
                    entry_price = None
                
                take_profit = float(decision_data.get('take_profit', 0)) if decision_data.get('take_profit') else None
                stop_loss = float(decision_data.get('stop_loss', 0)) if decision_data.get('stop_loss') else None
                confidence = float(decision_data.get('confidence', 50))
                reason = decision_data.get('reason', 'AI Decision')
                
                return direction, entry_price, take_profit, stop_loss, confidence, reason
            return 'HOLD', None, None, None, 50, 'No valid JSON found'
        except Exception as e:
            self.print_color(f"AI response parsing failed: {e}", self.Fore.RED)
            return 'HOLD', None, None, None, 50, 'Parsing failed'

    def get_deepseek_analysis(self, pair, market_data):
        try:
            if not self.openrouter_key:
                self.print_color("OpenRouter API key not found", self.Fore.RED)
                return "HOLD", None, None, None, 0, "No API key"
            
            current_price = market_data['current_price']
            price_change = market_data.get('price_change', 0)
            volume_change = market_data.get('volume_change', 0)
            
            # FULL AI CONTROL - FIXED $50 & 5x
            prompt = f"""
<|system|>
You are DeepSeek V3.1 in FULL AUTHORIZATION MODE.
You MUST use:
- Trade size: EXACTLY $50 USD
- Leverage: EXACTLY 5x
- Pairs: SOLUSDT, XRPUSDT, LINKUSDT, DOGEUSDT, SUIUSDT only

You decide:
- Direction (LONG/SHORT/HOLD)
- Entry price
- Take Profit
- Stop Loss
- Confidence (0-100)

Only trade if confidence >= 80%. Return PERFECT JSON only.
</|system|>

<|user|>
**FULL AI CONTROL 3-MINUTE SCALPING for {pair}**

**MARKET DATA:**
- Current Price: ${current_price:.6f}
- 15min Price Change: {price_change:.2f}%
- 15min Volume Change: {volume_change:.2f}%
- Recent Prices (last 8): {market_data.get('prices', [])[-8:]}
- Highs (last 5): {market_data.get('highs', [])[-5:]}
- Lows (last 5): {market_data.get('lows', [])[-5:]}

**YOU DECIDE:**
1. Analyze momentum, volume, patterns
2. Calculate exact entry, TP, SL
3. Use $50 and 5x leverage → quantity = (50 * 5) / entry_price
4. Only trade if confidence >= 80%

**RETURN ONLY THIS JSON:**
```json
{{
  "direction": "LONG" | "SHORT" | "HOLD",
  "entry_price": number,
  "take_profit": number,
  "stop_loss": number,
  "confidence": 0-100,
  "reason": "brief reason"
}}
"""
            headers = {
                "Authorization": f"Bearer {self.openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://your-bot.com",
                "X-Title": "AI $50 5x Scalper"
            }
            data = {
                "model": "deepseek/deepseek-chat-v3.1",
                "messages": [
                    {"role": "system", "content": "You are in FULL AUTHORIZATION MODE. Return JSON only."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 500,
                "top_p": 0.9
            }        self.print_color(f"AI Analyzing {pair}...", self.Fore.MAGENTA + self.Style.BRIGHT)
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=40)
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result['choices'][0]['message']['content'].strip()
            
            direction, entry_price, take_profit, stop_loss, confidence, reason = self.parse_ai_response(ai_response)
            
            if direction == "HOLD" or confidence < 80:
                self.print_color(f"HOLD {pair} ({confidence}% confidence)", self.Fore.YELLOW)
                return "HOLD", None, None, None, confidence, reason
            
            # Calculate quantity based on $50 and 5x
            quantity = (self.trade_size_usd * self.default_leverage) / entry_price
            quantity = self.format_quantity(pair, quantity)
            
            direction_icon = "LONG" if direction == "LONG" else "SHORT"
            color = self.Fore.GREEN + self.Style.BRIGHT if direction == "LONG" else self.Fore.RED + self.Style.BRIGHT
            
            self.print_color(f"{direction_icon} {pair} | Entry: ${entry_price:.4f} | Qty: {quantity} | Conf: {confidence}%", color)
            self.print_color(f"   AI TP: ${take_profit:.4f} | SL: ${stop_loss:.4f}", self.Fore.CYAN)
            self.print_color(f"   Reason: {reason}", self.Fore.YELLOW)
            
            return direction, entry_price, take_profit, stop_loss, quantity, confidence, reason
        else:
            self.print_color(f"API error: {response.status_code}", self.Fore.RED)
            return "HOLD", None, None, None, 0, "API Error"
            
    except Exception as e:
        self.print_color(f"AI analysis failed: {e}", self.Fore.RED)
        return "HOLD", None, None, None, 0, "Error"

def get_price_history(self, pair, limit=15):
    try:
        if self.binance:
            klines = self.binance.futures_klines(symbol=pair, interval=Client.KLINE_INTERVAL_3MINUTE, limit=limit)
            prices = [float(k[4]) for k in klines]
            highs =  [float(k[2]) for k in klines]
            lows =   [float(k[3]) for k in klines]
            volumes = [float(k[5]) for k in klines]
            
            current_price = prices[-1] if prices else 0
            price_change = ((current_price - prices[-6]) / prices[-6] * 100) if len(prices) >= 6 else 0
            volume_change = ((volumes[-1] - volumes[-6]) / volumes[-6] * 100) if len(volumes) >= 6 else 0
            
            return {
                'prices': prices, 
                'highs': highs,
                'lows': lows,
                'volumes': volumes,
                'current_price': current_price,
                'price_change': price_change,
                'volume_change': volume_change
            }
        else:
            current_price = self.get_current_price(pair)
            return {
                'prices': [current_price] * 10, 
                'highs': [current_price * 1.01] * 10,
                'lows': [current_price * 0.99] * 10,
                'volumes': [100000] * 10,
                'current_price': current_price,
                'price_change': 0.5,
                'volume_change': 10.2
            }
    except Exception as e:
        current_price = self.get_current_price(pair)
        return {
            'prices': [current_price] * 10,
            'highs': [current_price * 1.01] * 10,
            'lows': [current_price * 0.99] * 10,
            'volumes': [100000] * 10,
            'current_price': current_price,
            'price_change': 0.5,
            'volume_change': 10.2
        }

def get_ai_decision(self, pair_data):
    try:
        pair = list(pair_data.keys())[0]
        current_price = pair_data[pair]['price']
        if current_price <= 0:
            return {"action": "HOLD", "confidence": 0}
        
        market_data = self.get_price_history(pair)
        market_data['current_price'] = current_price
        
        direction, entry_price, take_profit, stop_loss, quantity, confidence, reason = self.get_deepseek_analysis(pair, market_data)
        
        if direction == "HOLD" or confidence < 80:
            return {"action": "HOLD", "confidence": confidence}
        else:
            return {
                "action": "TRADE",
                "pair": pair,
                "direction": direction,
                "entry_price": entry_price,
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "quantity": quantity,
                "confidence": confidence,
                "reason": reason
            }
    except Exception as e:
        self.print_color(f"AI decision failed: {e}", self.Fore.RED)
        return {"action": "HOLD", "confidence": 0}

def get_current_price(self, pair):
    try:
        if self.binance:
            ticker = self.binance.futures_symbol_ticker(symbol=pair)
            return float(ticker['price'])
        else:
            import requests
            url = f"https://api.binance.com/api/v3/ticker/price"
            params = {'symbol': pair}
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            return float(data['price'])
    except:
        base_prices = {
            "SOLUSDT": 180.50, "XRPUSDT": 0.62, "LINKUSDT": 18.75, 
            "DOGEUSDT": 0.12, "SUIUSDT": 2.45
        }
        return base_prices.get(pair, 100)

def get_market_data(self):
    market_data = {}
    for pair in self.available_pairs:
        try:
            price = self.get_current_price(pair)
            if price and price > 0:
                market_data[pair] = {'price': price}
        except Exception as e:
            continue
    return market_data

def can_open_new_trade(self, pair):
    if pair in self.bot_opened_trades and self.bot_opened_trades[pair]['status'] == 'ACTIVE':
        return False
    return len(self.bot_opened_trades) < self.max_concurrent_trades

def execute_trade(self, decision):
    try:
        pair = decision["pair"]
        if not self.can_open_new_trade(pair):
            self.print_color(f"Cannot open {pair} - max trades reached", self.Fore.RED)
            return False
        
        direction = decision["direction"]
        entry_price = decision["entry_price"]
        take_profit = decision["take_profit"]
        stop_loss = decision["stop_loss"]
        quantity = decision["quantity"]
        confidence = decision["confidence"]
        reason = decision["reason"]
        
        if not all([entry_price, take_profit, stop_loss, quantity]):
            self.print_color("Invalid AI parameters", self.Fore.RED)
            return False

        take_profit = self.format_price(pair, take_profit)
        stop_loss = self.format_price(pair, stop_loss)

        direction_color = self.Fore.GREEN + self.Style.BRIGHT if direction == 'LONG' else self.Fore.RED + self.Style.BRIGHT
        direction_icon = "LONG" if direction == 'LONG' else "SHORT"
        
        self.print_color(f"\nAI TRADE EXECUTION", self.Fore.CYAN + self.Style.BRIGHT)
        self.print_color("=" * 70, self.Fore.CYAN)
        self.print_color(f"{direction_icon} {pair} | $50 | 5x", direction_color)
        self.print_color(f"ENTRY: ${entry_price:.4f} | QTY: {quantity}", self.Fore.WHITE)
        self.print_color(f"TP: ${take_profit:.4f} | SL: ${stop_loss:.4f}", self.Fore.YELLOW)
        self.print_color(f"CONFIDENCE: {confidence}% | REASON: {reason}", self.Fore.MAGENTA + self.Style.BRIGHT)
        self.print_color("=" * 70, self.Fore.CYAN)
        
        entry_side = 'BUY' if direction == 'LONG' else 'SELL'
        try:
            order = self.binance.futures_create_order(
                symbol=pair,
                side=entry_side,
                type='MARKET',
                quantity=quantity
            )
            self.print_color(f"{direction} ORDER EXECUTED!", self.Fore.GREEN + self.Style.BRIGHT)
            time.sleep(1)
            
            stop_side = 'SELL' if direction == 'LONG' else 'BUY'
            self.binance.futures_create_order(
                symbol=pair, side=stop_side, type='STOP_MARKET',
                quantity=quantity, stopPrice=stop_loss, reduceOnly=True
            )
            self.binance.futures_create_order(
                symbol=pair, side=stop_side, type='TAKE_PROFIT_MARKET',
                quantity=quantity, stopPrice=take_profit, reduceOnly=True
            )
            
            self.bot_opened_trades[pair] = {
                "pair": pair, "direction": direction, "entry_price": entry_price,
                "quantity": quantity, "stop_loss": stop_loss, "take_profit": take_profit,
                "entry_time": time.time(), "status": 'ACTIVE',
                'ai_confidence': confidence, 'ai_reason': reason,
                'entry_time_th': self.get_thailand_time()
            }
            
            self.print_color(f"AI TRADE LIVE: {pair} {direction}", self.Fore.GREEN + self.Style.BRIGHT)
            return True
            
        except Exception as e:
            self.print_color(f"Execution Error: {e}", self.Fore.RED)
            return False
        
    except Exception as e:
        self.print_color(f"Trade failed: {e}", self.Fore.RED)
        return False

def get_live_position_data(self, pair):
    try:
        positions = self.binance.futures_position_information(symbol=pair)
        for pos in positions:
            if pos['symbol'] == pair and float(pos['positionAmt']) != 0:
                entry_price = float(pos.get('entryPrice', 0))
                quantity = abs(float(pos['positionAmt']))
                unrealized_pnl = float(pos.get('unRealizedProfit', 0))
                ticker = self.binance.futures_symbol_ticker(symbol=pair)
                current_price = float(ticker['price'])
                direction = "SHORT" if pos['positionAmt'].startswith('-') else "LONG"
                return {
                    'direction': direction,
                    'entry_price': entry_price,
                    'quantity': quantity,
                    'current_price': current_price,
                    'unrealized_pnl': unrealized_pnl,
                    'status': 'ACTIVE'
                }
        return None
    except Exception as e:
        self.print_color(f"Error getting live data: {e}", self.Fore.RED)
        return None

def monitor_positions(self):
    try:
        closed_trades = []
        for pair, trade in list(self.bot_opened_trades.items()):
            if trade['status'] != 'ACTIVE':
                continue
            
            live_data = self.get_live_position_data(pair)
            if not live_data:
                self.close_trade_with_cleanup(pair, trade, "AI EXIT")
                closed_trades.append(pair)
                continue
                
            direction_icon = "LONG" if trade['direction'] == 'LONG' else "SHORT"
            pnl_color = self.Fore.GREEN + self.Style.BRIGHT if live_data['unrealized_pnl'] >= 0 else self.Fore.RED + self.Style.BRIGHT
            
            self.print_color(f"\nLIVE: {pair} {direction_icon} | P&L: ${live_data['unrealized_pnl']:.2f}", pnl_color)
            self.print_color(f"   Entry: ${trade['entry_price']:.4f} | Current: ${live_data['current_price']:.4f}", self.Fore.WHITE)
            self.print_color(f"   TP: ${trade['take_profit']:.4f} | SL: ${trade['stop_loss']:.4f}", self.Fore.YELLOW)
                
        return closed_trades
    except Exception as e:
        self.print_color(f"Monitoring error: {e}", self.Fore.RED)
        return []

def close_trade_with_cleanup(self, pair, trade, close_reason="AI"):
    try:
        open_orders = self.binance.futures_get_open_orders(symbol=pair)
        canceled = 0
        for order in open_orders:
            if order['reduceOnly'] and order['symbol'] == pair:
                try:
                    self.binance.futures_cancel_order(symbol=pair, orderId=order['orderId'])
                    canceled += 1
                except: pass
        
        final_pnl = self.get_final_pnl(pair, trade)
        trade['status'] = 'CLOSED'
        trade['exit_time_th'] = self.get_thailand_time()
        trade['exit_price'] = self.get_current_price(pair)
        trade['pnl'] = final_pnl
        trade['close_reason'] = close_reason
        
        closed_trade = trade.copy()
        self.add_trade_to_history(closed_trade)
        
        pnl_color = self.Fore.GREEN + self.Style.BRIGHT if final_pnl > 0 else self.Fore.RED + self.Style.BRIGHT
        direction_icon = "LONG" if trade['direction'] == 'LONG' else "SHORT"
        self.print_color(f"\nAI CLOSED: {pair} {direction_icon}", pnl_color)
        self.print_color(f"   Final P&L: ${final_pnl:.2f}", pnl_color)
        if canceled > 0:
            self.print_color(f"   Cleaned {canceled} orders", self.Fore.CYAN)
            
        del self.bot_opened_trades[pair]
        
    except Exception as e:
        self.print_color(f"Cleanup failed: {e}", self.Fore.RED)

def get_final_pnl(self, pair, trade):
    try:
        live = self.get_live_position_data(pair)
        if live and 'unrealized_pnl' in live:
            return live['unrealized_pnl']
        current = self.get_current_price(pair)
        if not current:
            return 0
        if trade['direction'] == 'LONG':
            return (current - trade['entry_price']) * trade['quantity']
        else:
            return (trade['entry_price'] - current) * trade['quantity']
    except:
        return 0

def display_dashboard(self):
    self.print_color(f"\nAI DASHBOARD - {self.get_thailand_time()}", self.Fore.RED + self.Style.BRIGHT)
    self.print_color("=" * 90, self.Fore.RED)
    
    active_count = 0
    total_unrealized = 0
    
    for pair, trade in self.bot_opened_trades.items():
        if trade['status'] == 'ACTIVE':
            active_count += 1
            live_data = self.get_live_position_data(pair)
            if live_data:
                direction_icon = "LONG" if trade['direction'] == 'LONG' else "SHORT"
                pnl_color = self.Fore.GREEN + self.Style.BRIGHT if live_data['unrealized_pnl'] >= 0 else self.Fore.RED + self.Style.BRIGHT
                total_unrealized += live_data['unrealized_pnl']
                
                self.print_color(f"{direction_icon} {pair} | Qty: {trade['quantity']}", self.Fore.WHITE + self.Style.BRIGHT)
                self.print_color(f"   Entry: ${trade['entry_price']:.4f} | Current: ${live_data['current_price']:.4f}", self.Fore.WHITE)
                self.print_color(f"   P&L: ${live_data['unrealized_pnl']:.2f}", pnl_color)
                self.print_color(f"   TP: ${trade['take_profit']:.4f} | SL: ${trade['stop_loss']:.4f}", self.Fore.YELLOW)
                self.print_color("   " + "-" * 60, self.Fore.CYAN)
    
    if active_count == 0:
        self.print_color("No active AI positions", self.Fore.YELLOW)
    else:
        total_color = self.Fore.GREEN + self.Style.BRIGHT if total_unrealized >= 0 else self.Fore.RED + self.Style.BRIGHT
        self.print_color(f"Active: {active_count} | Unrealized P&L: ${total_unrealized:.2f}", total_color)

def run_trading_cycle(self):
    try:
        closed_trades = self.monitor_positions()
        self.display_dashboard()
        
        if hasattr(self, 'cycle_count') and self.cycle_count % 5 == 0:
            self.show_trade_history(8)
            self.show_trading_stats()
        
        market_data = self.get_market_data()
        if market_data:
            self.print_color(f"\nAI SCANNING {len(market_data)} PAIRS...", self.Fore.BLUE + self.Style.BRIGHT)
            
            qualified_signals = 0
            for pair in market_data.keys():
                if self.can_open_new_trade(pair):
                    pair_data = {pair: market_data[pair]}
                    decision = self.get_ai_decision(pair_data)
                    
                    if decision["action"] == "TRADE":
                        qualified_signals += 1
                        self.execute_trade(decision)
                        time.sleep(1)
                else:
                    if pair not in self.bot_opened_trades:
                        self.print_color(f"SKIPPED: {pair} (max trades)", self.Fore.MAGENTA)
            
            if qualified_signals == 0:
                self.print_color("No AI signals this cycle", self.Fore.YELLOW)
            else:
                self.print_color(f"{qualified_signals} AI signals executed", self.Fore.GREEN + self.Style.BRIGHT)
        else:
            self.print_color("No market data", self.Fore.RED)
            
    except Exception as e:
        self.print_color(f"Cycle error: {e}", self.Fore.RED)

def start_trading(self):
    self.print_color("FULL AI CONTROL BOT STARTED!", self.Fore.RED + self.Style.BRIGHT)
    self.print_color("FIXED: $50 | 5x | SOL XRP LINK DOGE SUI", self.Fore.CYAN + self.Style.BRIGHT)
    self.cycle_count = 0
    
    while True:
        try:
            self.cycle_count += 1
            self.print_color(f"\nAI CYCLE {self.cycle_count}", self.Fore.RED + self.Style.BRIGHT)
            self.print_color("=" * 60, self.Fore.RED)
            self.run_trading_cycle()
            self.print_color(f"Waiting 25s...", self.Fore.BLUE)
            time.sleep(25)
            
        except KeyboardInterrupt:
            self.print_color(f"\nAI BOT STOPPED", self.Fore.RED + self.Style.BRIGHT)
            self.show_trade_history(10)
            self.show_trading_stats()
            break
        except Exception as e:
            self.print_color(f"Error: {e}", self.Fore.RED)
            time.sleep(25)

    class AggressiveThreeMinPaperTradingBot:
    def init(self, real_bot):
        self.real_bot = real_bot
        self.Fore = real_bot.Fore
        self.Back = real_bot.Back
        self.Style = real_bot.Style
        self.COLORAMA_AVAILABLE = real_bot.COLORAMA_AVAILABLE    self.paper_balance = 5000
    self.paper_positions = {}
    self.paper_history = []
    
    self.real_bot.print_color("PAPER TRADING MODE ACTIVATED", self.Fore.GREEN + self.Style.BRIGHT)
    self.real_bot.print_color(f"Starting Balance: ${self.paper_balance}", self.Fore.CYAN + self.Style.BRIGHT)
    self.real_bot.print_color("AI FULL CONTROL - $50 | 5x | SOL XRP LINK DOGE SUI", self.Fore.CYAN + self.Style.BRIGHT)
    
def paper_execute_trade(self, decision):
    try:
        pair = decision["pair"]
        direction = decision["direction"]
        entry_price = decision["entry_price"]
        take_profit = decision["take_profit"]
        stop_loss = decision["stop_loss"]
        quantity = decision["quantity"]
        confidence = decision["confidence"]
        
        if pair in self.paper_positions:
            return False
            
        take_profit = self.real_bot.format_price(pair, take_profit)
        stop_loss = self.real_bot.format_price(pair, stop_loss)
        
        direction_color = self.Fore.GREEN + self.Style.BRIGHT if direction == 'LONG' else self.Fore.RED + self.Style.BRIGHT
        direction_icon = "LONG" if direction == 'LONG' else "SHORT"
        
        self.real_bot.print_color(f"\nPAPER TRADE", self.Fore.CYAN + self.Style.BRIGHT)
        self.print_color("=" * 70, self.Fore.CYAN)
        self.print_color(f"{direction_icon} {pair} | $50 | 5x", direction_color)
        self.print_color(f"ENTRY: ${entry_price:.4f} | QTY: {quantity}", self.Fore.WHITE)
        self.print_color(f"TP: ${take_profit:.4f} | SL: ${stop_loss:.4f}", self.Fore.YELLOW)
        self.print_color(f"CONF: {confidence}%", self.Fore.MAGENTA + self.Style.BRIGHT)
        self.print_color("=" * 70, self.Fore.CYAN)
        
        self.paper_positions[pair] = {
            "pair": pair, "direction": direction, "entry_price": entry_price,
            "quantity": quantity, "stop_loss": stop_loss, "take_profit": take_profit,
            "entry_time": time.time(), "status": 'ACTIVE',
            'entry_time_th': self.real_bot.get_thailand_time()
        }
        return True
    except Exception as e:
        self.real_bot.print_color(f"Paper trade failed: {e}", self.Fore.RED)
        return False

def monitor_paper_positions(self):
    try:
        closed = []
        for pair, trade in list(self.paper_positions.items()):
            if trade['status'] != 'ACTIVE':
                continue
            
            current_price = self.real_bot.get_current_price(pair)
            if not current_price:
                continue
            
            should_close = False
            close_reason = ""
            pnl = 0
            
            if trade['direction'] == 'LONG':
                if current_price >= trade['take_profit']:
                    should_close = True
                    close_reason = "TP"
                    pnl = (current_price - trade['entry_price']) * trade['quantity']
                elif current_price <= trade['stop_loss']:
                    should_close = True
                    close_reason = "SL"
                    pnl = (current_price - trade['entry_price']) * trade['quantity']
            else:
                if current_price <= trade['take_profit']:
                    should_close = True
                    close_reason = "TP"
                    pnl = (trade['entry_price'] - current_price) * trade['quantity']
                elif current_price >= trade['stop_loss']:
                    should_close = True
                    close_reason = "SL"
                    pnl = (trade['entry_price'] - current_price) * trade['quantity']
            
            if should_close:
                trade['status'] = 'CLOSED'
                trade['exit_price'] = current_price
                trade['pnl'] = pnl
                trade['close_reason'] = close_reason
                trade['close_time'] = self.real_bot.get_thailand_time()
                
                self.paper_balance += pnl
                self.paper_history.append(trade.copy())
                closed.append(pair)
                
                pnl_color = self.Fore.GREEN + self.Style.BRIGHT if pnl > 0 else self.Fore.RED + self.Style.BRIGHT
                direction_icon = "LONG" if trade['direction'] == 'LONG' else "SHORT"
                self.real_bot.print_color(f"\nPAPER CLOSED: {pair} {direction_icon}", pnl_color)
                self.real_bot.print_color(f"   P&L: ${pnl:.2f} | {close_reason}", pnl_color)
                self.real_bot.print_color(f"   Balance: ${self.paper_balance:.2f}", self.Fore.CYAN)
                
                del self.paper_positions[pair]
                
        return closed
    except Exception as e:
        self.real_bot.print_color(f"Paper monitoring error: {e}", self.Fore.RED)
        return []

def run_paper_cycle(self):
    try:
        self.monitor_paper_positions()
        market_data = self.real_bot.get_market_data()
        if market_data:
            for pair in market_data.keys():
                if pair not in self.paper_positions and len(self.paper_positions) < self.real_bot.max_concurrent_trades:
                    pair_data = {pair: market_data[pair]}
                    decision = self.real_bot.get_ai_decision(pair_data)
                    if decision["action"] == "TRADE":
                        self.paper_execute_trade(decision)
                        time.sleep(1)
    except Exception as e:
        self.real_bot.print_color(f"Paper cycle error: {e}", self.Fore.RED)

def start_paper_trading(self):
    self.real_bot.print_color("PAPER TRADING STARTED!", self.Fore.GREEN + self.Style.BRIGHT)
    cycle = 0
    while True:
        try:
            cycle += 1
            self.real_bot.print_color(f"\nPAPER CYCLE {cycle}", self.Fore.GREEN + self.Style.BRIGHT)
            self.run_paper_cycle()
            time.sleep(25)
        except KeyboardInterrupt:
            self.real_bot.print_color("PAPER TRADING STOPPED", self.Fore.YELLOW)
            breakif name == "main":
    try:
        bot = AggressiveThreeMinScalpingBot()    print("\n" + "="*80)
    print("AGGRESSIVE 3MIN AI SCALPING BOT - DEEPSEEK V3.1")
    print("="*80)
    print("SELECT MODE:")
    print("1. Live Trading ($50 | 5x | Real Money)")
    print("2. Paper Trading (No Risk)")
    
    choice = input("Enter choice (1-2): ").strip()
    
    if choice == "1":
        confirm = input("Type 'AGGRESSIVE' to confirm: ").strip()
        if confirm.upper() == 'AGGRESSIVE':
            bot.start_trading()
        else:
            print("Cancelled.")
    else:
        paper_bot = AggressiveThreeMinPaperTradingBot(bot)
        paper_bot.start_paper_trading()
        
except Exception as e:
    print(f"Failed to start bot: {e}")

        
