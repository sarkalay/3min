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

class SmartAITradingBot:
    def __init__(self):
        # Load config from .env file
        self.binance_api_key = os.getenv('BINANCE_API_KEY')
        self.binance_secret = os.getenv('BINANCE_SECRET_KEY')
        self.openrouter_key = os.getenv('OPENROUTER_API_KEY')
        
        # Store colorama references
        self.Fore = Fore
        self.Back = Back
        self.Style = Style
        self.COLORAMA_AVAILABLE = COLORAMA_AVAILABLE
        
        # Thailand timezone
        self.thailand_tz = pytz.timezone('Asia/Bangkok')
        
        # Trading Settings
        self.trade_size_usd = 50
        self.default_leverage = 5
        self.max_concurrent_trades = 3
        self.available_pairs = ["SOLUSDT", "XRPUSDT", "LINKUSDT", "DOGEUSDT", "SUIUSDT"]
        
        # AI decides everything
        self.required_confidence = 75
        
        # Track bot-opened trades only
        self.bot_opened_trades = {}
        
        # Trade history - Separate files for paper and real trading
        self.real_trade_history_file = "real_trading_history.json"
        self.paper_trade_history_file = "paper_trading_history.json"
        self.trade_history = self.load_trade_history()
        
        # Paper trading balance
        self.paper_balance = 10000  # Starting paper balance
        self.initial_balance = 10000
        
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
            self.trade_history_file = self.real_trade_history_file
            self.print_color("SMART AI TRADING BOT WITH DEEP THINKING ACTIVATED!", self.Fore.GREEN + self.Style.BRIGHT)
            self.print_color("💰 REAL TRADING MODE - Connected to Binance", self.Fore.CYAN + self.Style.BRIGHT)
        except Exception as e:
            self.binance = None
            self.trade_history_file = self.paper_trade_history_file
            self.print_color("SMART AI TRADING BOT WITH DEEP THINKING ACTIVATED!", self.Fore.GREEN + self.Style.BRIGHT)
            self.print_color("📝 PAPER TRADING MODE - No Binance connection", self.Fore.YELLOW + self.Style.BRIGHT)
            self.print_color(f"💰 Starting Paper Balance: ${self.paper_balance}", self.Fore.CYAN)
        
        self.print_color("AI DECIDES EVERYTHING: Entry, TP, SL, Analysis", self.Fore.MAGENTA + self.Style.BRIGHT)
        self.print_color("15M TIMEFRAME + DEEP THINKING ANALYSIS", self.Fore.BLUE + self.Style.BRIGHT)
        
        self.validate_config()
        if self.binance:
            self.setup_futures()
            self.load_symbol_precision()
    
    def load_trade_history(self):
        """Load trade history from appropriate file based on mode"""
        try:
            history_file = self.real_trade_history_file if self.binance else self.paper_trade_history_file
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history = json.load(f)
                    # Calculate statistics from loaded history
                    if history:
                        self.total_trades = len(history)
                        self.winning_trades = len([t for t in history if t.get('pnl', 0) > 0])
                        self.total_pnl = sum(t.get('pnl', 0) for t in history)
                    return history
            return []
        except Exception as e:
            self.print_color(f"Error loading trade history: {e}", self.Fore.RED)
            return []
    
    def save_trade_history(self):
        """Save trade history to appropriate file"""
        try:
            with open(self.trade_history_file, 'w') as f:
                json.dump(self.trade_history, f, indent=2)
            self.print_color(f"💾 Trade history saved to {self.trade_history_file}", self.Fore.CYAN)
        except Exception as e:
            self.print_color(f"Error saving trade history: {e}", self.Fore.RED)
    
    def add_trade_to_history(self, trade_data):
        """Add trade to history and update statistics"""
        try:
            trade_data['close_time'] = self.get_thailand_time()
            trade_data['close_timestamp'] = time.time()
            trade_data['mode'] = 'REAL' if self.binance else 'PAPER'
            
            self.trade_history.append(trade_data)
            
            # Update statistics
            self.total_trades += 1
            pnl = trade_data.get('pnl', 0)
            self.total_pnl += pnl
            if pnl > 0:
                self.winning_trades += 1
            
            # Update paper balance if in paper mode
            if not self.binance:
                self.paper_balance += pnl
                trade_data['paper_balance_after'] = self.paper_balance
            
            # Keep only last 200 trades
            if len(self.trade_history) > 200:
                self.trade_history = self.trade_history[-200:]
            
            self.save_trade_history()
            
            # Print trade result
            pnl_color = self.Fore.GREEN if pnl > 0 else self.Fore.RED if pnl < 0 else self.Fore.YELLOW
            self.print_color(f"💾 Trade saved to history: {trade_data['pair']} {trade_data['direction']} P&L: ${pnl:.2f}", pnl_color)
            if not self.binance:
                self.print_color(f"💰 Paper Balance: ${self.paper_balance:.2f}", self.Fore.CYAN)
                
        except Exception as e:
            self.print_color(f"Error adding trade to history: {e}", self.Fore.RED)
    
    def show_trading_stats(self):
        """Show trading statistics"""
        if self.total_trades == 0:
            self.print_color("No trades yet", self.Fore.YELLOW)
            return
            
        win_rate = (self.winning_trades / self.total_trades) * 100
        avg_trade = self.total_pnl / self.total_trades
        
        self.print_color(f"\n📊 TRADING STATISTICS", self.Fore.CYAN + self.Style.BRIGHT)
        self.print_color("=" * 50, self.Fore.CYAN)
        self.print_color(f"Total Trades: {self.total_trades}", self.Fore.WHITE)
        self.print_color(f"Winning Trades: {self.winning_trades}", self.Fore.GREEN)
        self.print_color(f"Win Rate: {win_rate:.1f}%", self.Fore.GREEN + self.Style.BRIGHT if win_rate > 50 else self.Fore.YELLOW)
        self.print_color(f"Total P&L: ${self.total_pnl:.2f}", self.Fore.GREEN + self.Style.BRIGHT if self.total_pnl > 0 else self.Fore.RED + self.Style.BRIGHT)
        self.print_color(f"Average P&L per Trade: ${avg_trade:.2f}", self.Fore.WHITE)
        
        if not self.binance:
            balance_change = ((self.paper_balance - self.initial_balance) / self.initial_balance) * 100
            change_color = self.Fore.GREEN if balance_change >= 0 else self.Fore.RED
            self.print_color(f"💰 Paper Balance: ${self.paper_balance:.2f}", self.Fore.CYAN)
            self.print_color(f"📈 Balance Change: {balance_change:+.2f}%", change_color)
    
    def show_trade_history(self, limit=10):
        """Show recent trade history"""
        if not self.trade_history:
            self.print_color("No trade history found", self.Fore.YELLOW)
            return
        
        self.print_color(f"\n📋 RECENT TRADE HISTORY (Last {min(limit, len(self.trade_history))} trades)", self.Fore.MAGENTA + self.Style.BRIGHT)
        self.print_color("=" * 80, self.Fore.MAGENTA)
        
        recent_trades = self.trade_history[-limit:]
        for i, trade in enumerate(reversed(recent_trades)):
            pnl = trade.get('pnl', 0)
            pnl_color = self.Fore.GREEN + self.Style.BRIGHT if pnl > 0 else self.Fore.RED + self.Style.BRIGHT if pnl < 0 else self.Fore.YELLOW
            direction_icon = "🟢 LONG" if trade['direction'] == 'LONG' else "🔴 SHORT"
            close_reason = trade.get('close_reason', 'N/A')
            mode = trade.get('mode', 'UNKNOWN')
            
            self.print_color(f"{i+1:2d}. {direction_icon} {trade['pair']} | Entry: ${trade.get('entry_price', 0):.4f} | Exit: ${trade.get('exit_price', 0):.4f}", pnl_color)
            self.print_color(f"     P&L: ${pnl:.2f} | {close_reason} | {mode}", pnl_color)
    
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
        if not self.openrouter_key:
            self.print_color("Missing OpenRouter API key!", self.Fore.RED)
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
                except Exception as e:
                    pass
            self.print_color("✅ Futures setup completed", self.Fore.GREEN)
        except Exception as e:
            self.print_color(f"Futures setup failed: {e}", self.Fore.RED)
    
    def load_symbol_precision(self):
        if not self.binance:
            for pair in self.available_pairs:
                self.quantity_precision[pair] = 3
                self.price_precision[pair] = 4
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
            self.print_color("✅ Symbol precision loaded", self.Fore.GREEN)
        except Exception as e:
            for pair in self.available_pairs:
                self.quantity_precision[pair] = 3
                self.price_precision[pair] = 4
    
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
            return 'HOLD', None, None, None, 50, 'Parsing failed'

    def get_klines_data(self, pair, interval='15m', limit=20):
        """Get Kline data from Binance"""
        try:
            if self.binance:
                klines = self.binance.futures_klines(symbol=pair, interval=interval, limit=limit)
            else:
                # Fallback to REST API
                url = "https://api.binance.com/api/v3/klines"
                params = {
                    'symbol': pair,
                    'interval': interval,
                    'limit': limit
                }
                response = requests.get(url, params=params, timeout=10)
                klines = response.json()
            
            formatted_klines = []
            for k in klines:
                formatted_klines.append({
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5])
                })
            return formatted_klines
        except Exception as e:
            self.print_color(f"Error getting klines for {pair}: {e}", self.Fore.RED)
            return None

    def get_market_analysis(self, pair):
        """Get comprehensive market analysis for a pair"""
        try:
            klines_15m = self.get_klines_data(pair, '15m', 20)
            
            if not klines_15m:
                return None
            
            current_price = klines_15m[-1]['close']
            
            # Calculate basic technical levels
            highs_15m = [k['high'] for k in klines_15m]
            lows_15m = [k['low'] for k in klines_15m]
            closes_15m = [k['close'] for k in klines_15m]
            
            # Support and Resistance
            recent_high = max(highs_15m[-5:])
            recent_low = min(lows_15m[-5:])
            
            # Trend analysis
            ma_5 = sum(closes_15m[-5:]) / 5
            ma_10 = sum(closes_15m[-10:]) / 10
            trend = "BULLISH" if ma_5 > ma_10 else "BEARISH" if ma_5 < ma_10 else "SIDEWAYS"
            
            # Price changes
            price_change_15m = ((current_price - closes_15m[-2]) / closes_15m[-2]) * 100
            
            analysis = {
                'pair': pair,
                'current_price': current_price,
                'trend_15m': trend,
                'recent_high': recent_high,
                'recent_low': recent_low,
                'price_change_15m': price_change_15m,
                'support_levels': [recent_low, min(lows_15m[-10:])],
                'resistance_levels': [recent_high, max(highs_15m[-10:])],
                'volume_trend': 'increasing' if klines_15m[-1]['volume'] > klines_15m[-2]['volume'] else 'decreasing'
            }
            
            return analysis
        except Exception as e:
            self.print_color(f"Market analysis failed for {pair}: {e}", self.Fore.RED)
            return None

    def get_ai_trading_decision(self, pair):
        """AI analyzes market and makes trading decision with DEEP THINKING"""
        try:
            if not self.openrouter_key:
                return "HOLD", None, None, None, 0, "No API key"
            
            market_analysis = self.get_market_analysis(pair)
            if not market_analysis:
                return "HOLD", None, None, None, 0, "Market analysis failed"
            
            current_price = market_analysis['current_price']
            
            prompt = f"""
<|system|>
You are an expert cryptocurrency trader analyzing the 15-minute timeframe with DEEP THINKING process.

**DEEP THINKING PROCESS REQUIRED:**

**STEP 1: MARKET STRUCTURE ANALYSIS**
- Analyze the current trend direction
- Identify key support and resistance levels
- Assess market momentum and volume
- Determine if market is trending or ranging

**STEP 2: PRICE ACTION EVALUATION**
- Analyze candlestick patterns
- Look for confirmation signals
- Check for divergence/convergence
- Evaluate entry timing

**STEP 3: RISK ASSESSMENT**
- Calculate optimal position size
- Determine safe stop loss level
- Identify realistic take profit target
- Assess risk-reward ratio

**STEP 4: TRADE EXECUTION PLAN**
- Define exact entry price
- Set logical TP and SL levels
- Determine position size
- Establish confidence level

**TRADE PARAMETERS:**
- Trade Size: $50 with 5x leverage
- Focus on 15-minute timeframe
- Use proper risk management
- Be patient with entries

Return ONLY JSON format below after completing all thinking steps:
</|system|>

<|user|>
**DEEP THINKING ANALYSIS - {pair}**

**CURRENT MARKET DATA:**
- Current Price: ${current_price:.4f}
- 15M Trend: {market_analysis['trend_15m']}
- Price Change (15M): {market_analysis['price_change_15m']:.2f}%
- Recent High: ${market_analysis['recent_high']:.4f}
- Recent Low: ${market_analysis['recent_low']:.4f}
- Support Levels: {[f'${level:.4f}' for level in market_analysis['support_levels']]}
- Resistance Levels: {[f'${level:.4f}' for level in market_analysis['resistance_levels']]}
- Volume Trend: {market_analysis['volume_trend']}

**DEEP THINKING PROCESS:**

**STEP 1 - Market Structure:**
[Analyze the overall market structure on 15M timeframe. Is it bullish, bearish, or ranging?]

**STEP 2 - Price Action:**
[Evaluate current price action. Any patterns? Momentum? Confirmation signals?]

**STEP 3 - Key Levels:**
[Identify the most important support and resistance levels for TP/SL placement]

**STEP 4 - Risk Management:**
[Calculate optimal SL distance and TP targets based on market structure]

**STEP 5 - Final Decision:**
[Based on above analysis, make final trading decision with exact levels]

**FINAL TRADING DECISION:**
```json
{{
  "direction": "LONG" | "SHORT" | "HOLD",
  "entry_price": number,
  "take_profit": number,
  "stop_loss": number,
  "confidence": 0-100,
  "reason": "detailed reasoning based on deep thinking analysis"
}}
"""
            headers = {
                "Authorization": f"Bearer {self.openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://smart-ai-trader.com",
                "X-Title": "Smart AI Trader - Deep Thinking"
            }
            data = {
                "model": "deepseek/deepseek-chat-v3.1",
                "messages": [
                    {"role": "system", "content": "You are a professional trader using DEEP THINKING process. Return JSON only after complete analysis."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 1000,
                "top_p": 0.8
            }
            
            self.print_color(f"🧠 DEEP THINKING ANALYSIS: {pair}...", self.Fore.MAGENTA + self.Style.BRIGHT)
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=90)
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['choices'][0]['message']['content'].strip()
                
                direction, entry_price, take_profit, stop_loss, confidence, reason = self.parse_ai_response(ai_response)
                
                if direction == "HOLD" or confidence < self.required_confidence:
                    self.print_color(f"➖ HOLD {pair} ({confidence}% confidence)", self.Fore.YELLOW)
                    self.print_color(f"   Reason: {reason}", self.Fore.WHITE)
                    return "HOLD", None, None, None, confidence, reason
                
                # Calculate quantity
                quantity = (self.trade_size_usd * self.default_leverage) / entry_price
                quantity = self.format_quantity(pair, quantity)
                
                # Format prices
                take_profit = self.format_price(pair, take_profit)
                stop_loss = self.format_price(pair, stop_loss)
                
                direction_icon = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"
                color = self.Fore.GREEN + self.Style.BRIGHT if direction == "LONG" else self.Fore.RED + self.Style.BRIGHT
                
                # Calculate risk reward
                if direction == "LONG":
                    risk_reward = (take_profit - entry_price) / (entry_price - stop_loss) if entry_price > stop_loss else 0
                else:
                    risk_reward = (entry_price - take_profit) / (stop_loss - entry_price) if stop_loss > entry_price else 0
                
                self.print_color(f"✅ {direction_icon} {pair} - DEEP THINKING ANALYSIS", color)
                self.print_color(f"   📍 Entry: ${entry_price:.4f} | 📦 Qty: {quantity}", self.Fore.WHITE)
                self.print_color(f"   🎯 TP: ${take_profit:.4f} | 🛑 SL: ${stop_loss:.4f}", self.Fore.CYAN)
                self.print_color(f"   📊 R/R: 1:{risk_reward:.1f} | Confidence: {confidence}%", self.Fore.MAGENTA)
                self.print_color(f"   🧠 Analysis: {reason}", self.Fore.YELLOW)
                
                return direction, entry_price, take_profit, stop_loss, quantity, confidence, reason
            else:
                self.print_color(f"❌ API error: {response.status_code}", self.Fore.RED)
                return "HOLD", None, None, None, 0, "API Error"
                
        except Exception as e:
            self.print_color(f"❌ Deep thinking analysis failed: {e}", self.Fore.RED)
            return "HOLD", None, None, None, 0, "Error"

    def get_current_price(self, pair):
        try:
            if self.binance:
                ticker = self.binance.futures_symbol_ticker(symbol=pair)
                return float(ticker['price'])
            else:
                url = f"https://api.binance.com/api/v3/ticker/price"
                params = {'symbol': pair}
                response = requests.get(url, params=params, timeout=5)
                data = response.json()
                return float(data['price'])
        except:
            # Fallback prices
            base_prices = {
                "SOLUSDT": 146.25, "XRPUSDT": 0.62, "LINKUSDT": 18.75, 
                "DOGEUSDT": 0.12, "SUIUSDT": 2.45
            }
            return base_prices.get(pair, 100)

    def can_open_new_trade(self, pair):
        if pair in self.bot_opened_trades and self.bot_opened_trades[pair]['status'] == 'ACTIVE':
            return False
        return len(self.bot_opened_trades) < self.max_concurrent_trades

    def execute_trade(self, pair, direction, entry_price, take_profit, stop_loss, quantity, confidence, reason):
        try:
            if not self.can_open_new_trade(pair):
                self.print_color(f"⏭️  Max trades reached, skipping {pair}", self.Fore.YELLOW)
                return False
            
            if not all([entry_price, take_profit, stop_loss, quantity]):
                self.print_color("❌ Invalid trade parameters", self.Fore.RED)
                return False

            direction_color = self.Fore.GREEN + self.Style.BRIGHT if direction == 'LONG' else self.Fore.RED + self.Style.BRIGHT
            direction_icon = "🟢 LONG" if direction == 'LONG' else "🔴 SHORT"
            
            # Calculate risk reward
            if direction == "LONG":
                risk_reward = (take_profit - entry_price) / (entry_price - stop_loss) if entry_price > stop_loss else 0
            else:
                risk_reward = (entry_price - take_profit) / (stop_loss - entry_price) if stop_loss > entry_price else 0
            
            self.print_color(f"\n🚀 EXECUTING TRADE", self.Fore.CYAN + self.Style.BRIGHT)
            self.print_color("=" * 60, self.Fore.CYAN)
            self.print_color(f"{direction_icon} {pair}", direction_color)
            self.print_color(f"📍 Entry: ${entry_price:.4f} | 📦 Size: {quantity}", self.Fore.WHITE)
            self.print_color(f"🎯 Take Profit: ${take_profit:.4f}", self.Fore.GREEN)
            self.print_color(f"🛑 Stop Loss: ${stop_loss:.4f}", self.Fore.RED)
            self.print_color(f"📊 Risk/Reward: 1:{risk_reward:.1f}", self.Fore.MAGENTA)
            self.print_color(f"💡 Reason: {reason}", self.Fore.YELLOW)
            self.print_color("=" * 60, self.Fore.CYAN)
            
            if not self.binance:
                # Paper trading
                self.print_color("📝 PAPER TRADE EXECUTED", self.Fore.GREEN)
                self.bot_opened_trades[pair] = {
                    "pair": pair, "direction": direction, "entry_price": entry_price,
                    "quantity": quantity, "stop_loss": stop_loss, "take_profit": take_profit,
                    "entry_time": time.time(), "status": 'ACTIVE',
                    'ai_confidence': confidence, 'ai_reason': reason,
                    'entry_time_th': self.get_thailand_time(),
                    'paper_balance_before': self.paper_balance
                }
                return True
            
            # Real trading
            entry_side = 'BUY' if direction == 'LONG' else 'SELL'
            try:
                order = self.binance.futures_create_order(
                    symbol=pair,
                    side=entry_side,
                    type='MARKET',
                    quantity=quantity
                )
                
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
                
                self.print_color(f"✅ TRADE EXECUTED SUCCESSFULLY", self.Fore.GREEN + self.Style.BRIGHT)
                return True
                
            except Exception as e:
                self.print_color(f"❌ Trade execution failed: {e}", self.Fore.RED)
                return False
            
        except Exception as e:
            self.print_color(f"❌ Trade failed: {e}", self.Fore.RED)
            return False

    def monitor_positions(self):
        try:
            closed_trades = []
            for pair, trade in list(self.bot_opened_trades.items()):
                if trade['status'] != 'ACTIVE':
                    continue
                
                current_price = self.get_current_price(pair)
                if not current_price:
                    continue
                
                should_close = False
                close_reason = ""
                pnl = 0
                
                if trade['direction'] == 'LONG':
                    if current_price >= trade['take_profit']:
                        should_close = True
                        close_reason = "🎯 TP HIT"
                        pnl = (current_price - trade['entry_price']) * trade['quantity']
                    elif current_price <= trade['stop_loss']:
                        should_close = True
                        close_reason = "🛑 SL HIT"
                        pnl = (current_price - trade['entry_price']) * trade['quantity']
                else:
                    if current_price <= trade['take_profit']:
                        should_close = True
                        close_reason = "🎯 TP HIT"
                        pnl = (trade['entry_price'] - current_price) * trade['quantity']
                    elif current_price >= trade['stop_loss']:
                        should_close = True
                        close_reason = "🛑 SL HIT"
                        pnl = (trade['entry_price'] - current_price) * trade['quantity']
                
                if should_close:
                    trade['status'] = 'CLOSED'
                    trade['exit_price'] = current_price
                    trade['pnl'] = pnl
                    trade['close_reason'] = close_reason
                    trade['close_time'] = self.get_thailand_time()
                    
                    # Add to history
                    self.add_trade_to_history(trade.copy())
                    
                    closed_trades.append(pair)
                    
                    pnl_color = self.Fore.GREEN + self.Style.BRIGHT if pnl > 0 else self.Fore.RED + self.Style.BRIGHT
                    direction_icon = "LONG" if trade['direction'] == 'LONG' else "SHORT"
                    self.print_color(f"🔚 TRADE CLOSED: {pair} {direction_icon} - {close_reason}", pnl_color)
                    self.print_color(f"   💰 P&L: ${pnl:.2f}", pnl_color)
                    
                    del self.bot_opened_trades[pair]
                    
            return closed_trades
        except Exception as e:
            self.print_color(f"❌ Monitoring error: {e}", self.Fore.RED)
            return []

    def display_status(self):
        self.print_color(f"\n📊 TRADING STATUS - {self.get_thailand_time()}", self.Fore.CYAN + self.Style.BRIGHT)
        self.print_color("=" * 50, self.Fore.CYAN)
        
        active_count = len(self.bot_opened_trades)
        
        if active_count == 0:
            self.print_color("No active positions", self.Fore.YELLOW)
        else:
            for pair, trade in self.bot_opened_trades.items():
                if trade['status'] == 'ACTIVE':
                    current_price = self.get_current_price(pair)
                    if trade['direction'] == 'LONG':
                        pnl = (current_price - trade['entry_price']) * trade['quantity']
                    else:
                        pnl = (trade['entry_price'] - current_price) * trade['quantity']
                    
                    pnl_color = self.Fore.GREEN + self.Style.BRIGHT if pnl >= 0 else self.Fore.RED + self.Style.BRIGHT
                    direction_icon = "🟢 LONG" if trade['direction'] == 'LONG' else "🔴 SHORT"
                    
                    self.print_color(f"{direction_icon} {pair}", self.Fore.WHITE)
                    self.print_color(f"   📍 Entry: ${trade['entry_price']:.4f} | Current: ${current_price:.4f}", self.Fore.WHITE)
                    self.print_color(f"   💰 P&L: ${pnl:.2f}", pnl_color)
                    self.print_color(f"   🎯 TP: ${trade['take_profit']:.4f} | 🛑 SL: ${trade['stop_loss']:.4f}", self.Fore.CYAN)

    def run_trading_cycle(self):
        try:
            self.print_color(f"\n🔄 TRADING CYCLE STARTED", self.Fore.BLUE + self.Style.BRIGHT)
            
            # Monitor existing positions
            closed_trades = self.monitor_positions()
            if closed_trades:
                self.print_color(f"📈 Closed {len(closed_trades)} trades", self.Fore.GREEN)
            
            # Display current status
            self.display_status()
            
            # Show statistics every 5 cycles
            if hasattr(self, 'cycle_count') and self.cycle_count % 5 == 0:
                self.show_trading_stats()
                self.show_trade_history(5)
            
            # Look for new trading opportunities
            self.print_color(f"\n🔍 SCANNING FOR NEW SETUPS...", self.Fore.MAGENTA + self.Style.BRIGHT)
            
            signals_found = 0
            for pair in self.available_pairs:
                if self.can_open_new_trade(pair):
                    direction, entry_price, take_profit, stop_loss, quantity, confidence, reason = self.get_ai_trading_decision(pair)
                    
                    if direction != "HOLD" and confidence >= self.required_confidence:
                        signals_found += 1
                        self.execute_trade(pair, direction, entry_price, take_profit, stop_loss, quantity, confidence, reason)
                        time.sleep(2)  # Small delay between executions
            
            if signals_found == 0:
                self.print_color("💤 No trading signals this cycle", self.Fore.YELLOW)
            else:
                self.print_color(f"✅ Executed {signals_found} new trades", self.Fore.GREEN + self.Style.BRIGHT)
                
        except Exception as e:
            self.print_color(f"❌ Trading cycle error: {e}", self.Fore.RED)

    def start_trading(self):
        mode = "REAL" if self.binance else "PAPER"
        self.print_color(f"🚀 SMART AI TRADING BOT WITH DEEP THINKING STARTED!", self.Fore.GREEN + self.Style.BRIGHT)
        self.print_color(f"💰 Trade Size: $50 | Leverage: 5x | Mode: {mode}", self.Fore.CYAN)
        self.print_color("📈 Timeframe: 15 Minutes | AI Controlled", self.Fore.MAGENTA)
        self.print_color("🎯 Pairs: SOL, XRP, LINK, DOGE, SUI", self.Fore.YELLOW)
        self.print_color("🧠 Mode: DEEP THINKING ANALYSIS", self.Fore.BLUE + self.Style.BRIGHT)
        
        if not self.binance:
            self.print_color(f"💰 Starting Paper Balance: ${self.paper_balance}", self.Fore.CYAN)
        
        self.cycle_count = 0
        while True:
            try:
                self.cycle_count += 1
                self.print_color(f"\n🔄 CYCLE {self.cycle_count}", self.Fore.BLUE + self.Style.BRIGHT)
                self.print_color("=" * 40, self.Fore.BLUE)
                
                self.run_trading_cycle()
                
                self.print_color(f"\n⏰ Next analysis in 3 minutes...", self.Fore.WHITE)
                time.sleep(180)  # 3 minutes between cycles
                
            except KeyboardInterrupt:
                self.print_color(f"\n🛑 TRADING BOT STOPPED", self.Fore.RED + self.Style.BRIGHT)
                self.show_trading_stats()
                self.show_trade_history(10)
                break
            except Exception as e:
                self.print_color(f"❌ Error: {e}", self.Fore.RED)
                time.sleep(180)


if __name__ == "__main__":
    try:
        bot = SmartAITradingBot()
        
        print("\n" + "="*60)
        print("🤖 SMART AI TRADING BOT WITH DEEP THINKING")
        print("📊 15-MINUTE TIMEFRAME | DEEP THINKING ANALYSIS")
        print("🎯 AI CONTROLS: Entry, TP, SL, Analysis")
        print("💾 AUTO-SAVE: Paper & Real Trading History")
        print("="*60)
        
        bot.start_trading()
            
    except Exception as e:
        print(f"❌ Failed to start bot: {e}")
