import os
import requests
import json
import time
import re
import math
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pytz

# Colorama setup
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = True
    class DummyColors:
        def __getattr__(self, name):
            return ''
    Fore = DummyColors()
    Back = DummyColors() 
    Style = DummyColors()

# Load environment variables
load_dotenv()

class AggressiveAIScalpingBot:
    def __init__(self):
        # Load config from .env file
        self.binance_api_key = os.getenv('BINANCE_API_KEY')
        self.binance_secret = os.getenv('BINANCE_SECRET_KEY')
        self.deepseek_key = os.getenv('DEEPSEEK_API_KEY')
        
        # Store colorama references
        self.Fore = Fore
        self.Back = Back
        self.Style = Style
        self.COLORAMA_AVAILABLE = COLORAMA_AVAILABLE
        
        # Thailand timezone
        self.thailand_tz = pytz.timezone('Asia/Bangkok')
        
        # AGGRESSIVE AI PARAMETERS - AI decides everything
        self.trade_size_usd = 50  # Fixed $50 per trade
        self.leverage = 5         # Fixed 5x leverage
        
        # AGGRESSIVE MODE SETTINGS
        self.max_concurrent_trades = 5
        self.available_pairs = ["SOLUSDT", "AVAXUSDT", "XRPUSDT", "LINKUSDT", "DOTUSDT", "ADAUSDT", "MATICUSDT"]
        self.confidence_threshold = 65
        
        # Track bot-opened trades only
        self.bot_opened_trades = {}
        
        # Trade history
        self.trade_history_file = "aggressive_ai_trading_history.json"
        self.trade_history = self.load_trade_history()
        
        # Precision settings
        self.quantity_precision = {}
        self.price_precision = {}
        
        # Initialize Binance client
        try:
            self.binance = Client(self.binance_api_key, self.binance_secret)
            self.print_color(f"🔥 AGGRESSIVE AI TRADING BOT ACTIVATED!", self.Fore.RED + self.Style.BRIGHT)
        except Exception as e:
            self.print_color(f"Binance initialization failed: {e}", self.Fore.RED)
            self.binance = None
        
        self.validate_config()
        if self.binance:
            self.setup_futures()
            self.load_symbol_precision()

    def show_all_positions_dashboard(self):
        """လက်ရှိဝင်ထားသမျှ positions အားလုံးကို real-time dashboard"""
        try:
            self.print_color(f"\n🎯 LIVE TRADING DASHBOARD - {self.get_thailand_time()}", self.Fore.CYAN + self.Style.BRIGHT)
            self.print_color("=" * 120, self.Fore.CYAN)
            
            # Show all active positions from Binance
            all_positions = self.binance.futures_position_information()
            active_positions = [pos for pos in all_positions if float(pos['positionAmt']) != 0]
            
            if not active_positions:
                self.print_color("📭 No active positions", self.Fore.YELLOW)
                return
            
            total_unrealized_pnl = 0
            total_initial_margin = 0
            
            for pos in active_positions:
                position_amt = float(pos['positionAmt'])
                pair = pos['symbol']
                entry_price = float(pos['entryPrice'])
                leverage = int(pos['leverage'])
                unrealized_pnl = float(pos['unRealizedProfit'])
                initial_margin = float(pos.get('initialMargin', 0))
                
                total_unrealized_pnl += unrealized_pnl
                total_initial_margin += initial_margin
                
                position_side = "LONG" if position_amt > 0 else "SHORT"
                quantity = abs(position_amt)
                
                # Get current price
                current_price = self.get_current_price(pair)
                
                # Calculate percentages
                if entry_price > 0:
                    if position_side == "LONG":
                        pnl_percent = ((current_price - entry_price) / entry_price) * 100 * leverage
                        price_change = ((current_price - entry_price) / entry_price) * 100
                    else:
                        pnl_percent = ((entry_price - current_price) / entry_price) * 100 * leverage
                        price_change = ((entry_price - current_price) / entry_price) * 100
                else:
                    pnl_percent = 0
                    price_change = 0
                
                # Get TP/SL from bot tracking if available
                tp_price = None
                sl_price = None
                if pair in self.bot_opened_trades:
                    tp_price = self.bot_opened_trades[pair].get('take_profit')
                    sl_price = self.bot_opened_trades[pair].get('stop_loss')
                
                # Display position info
                direction_icon = "🟢" if position_side == "LONG" else "🔴"
                pnl_color = self.Fore.GREEN if unrealized_pnl >= 0 else self.Fore.RED
                price_color = self.Fore.GREEN if price_change >= 0 else self.Fore.RED
                
                self.print_color(f"{direction_icon} {pair} {position_side}", self.Fore.WHITE + self.Style.BRIGHT)
                self.print_color(f"   📊 Size: {quantity} | ⚡ {leverage}x | 💰 Margin: ${initial_margin:.2f}", self.Fore.WHITE)
                self.print_color(f"   📈 Entry: ${entry_price:.4f} | Current: ${current_price:.4f} ({price_change:+.2f}%)", price_color)
                self.print_color(f"   💸 P&L: ${unrealized_pnl:.2f} ({pnl_percent:+.2f}%)", pnl_color)
                
                if tp_price:
                    tp_distance = abs(tp_price - current_price) / current_price * 100
                    self.print_color(f"   🎯 TP: ${tp_price:.4f} ({tp_distance:.2f}% away)", self.Fore.GREEN)
                
                if sl_price:
                    sl_distance = abs(sl_price - current_price) / current_price * 100
                    self.print_color(f"   🛡️  SL: ${sl_price:.4f} ({sl_distance:.2f}% away)", self.Fore.RED)
                
                self.print_color("   " + "-" * 100, self.Fore.CYAN)
            
            # Show summary
            total_pnl_color = self.Fore.GREEN if total_unrealized_pnl >= 0 else self.Fore.RED
            self.print_color(f"📊 SUMMARY: {len(active_positions)} Positions | Total P&L: ${total_unrealized_pnl:.2f}", total_pnl_color)
            
        except Exception as e:
            self.print_color(f"❌ Error getting positions dashboard: {e}", self.Fore.RED)

    def show_bot_managed_positions(self):
        """Bot ကဖွင့်ထားတဲ့ positions တွေကိုပဲကြည့်ရန်"""
        try:
            active_bot_trades = [t for t in self.bot_opened_trades.values() if t['status'] == 'ACTIVE']
            
            if not active_bot_trades:
                return
            
            self.print_color(f"\n🤖 BOT-MANAGED POSITIONS", self.Fore.MAGENTA + self.Style.BRIGHT)
            self.print_color("=" * 100, self.Fore.MAGENTA)
            
            for trade in active_bot_trades:
                pair = trade['pair']
                live_data = self.get_live_position_data(pair)
                
                if live_data:
                    direction_icon = "🟢" if trade['direction'] == 'LONG' else "🔴"
                    pnl_color = self.Fore.GREEN if live_data['unrealized_pnl'] >= 0 else self.Fore.RED
                    
                    # Calculate percentages
                    entry_price = trade['entry_price']
                    current_price = live_data['current_price']
                    leverage = self.leverage
                    
                    if trade['direction'] == 'LONG':
                        pnl_percent = ((current_price - entry_price) / entry_price) * 100 * leverage
                        tp_percent = ((trade['take_profit'] - entry_price) / entry_price) * 100 * leverage
                        sl_percent = ((trade['stop_loss'] - entry_price) / entry_price) * 100 * leverage
                        tp_distance = ((trade['take_profit'] - current_price) / current_price) * 100
                        sl_distance = ((current_price - trade['stop_loss']) / current_price) * 100
                    else:
                        pnl_percent = ((entry_price - current_price) / entry_price) * 100 * leverage
                        tp_percent = ((entry_price - trade['take_profit']) / entry_price) * 100 * leverage
                        sl_percent = ((entry_price - trade['stop_loss']) / entry_price) * 100 * leverage
                        tp_distance = ((current_price - trade['take_profit']) / current_price) * 100
                        sl_distance = ((trade['stop_loss'] - current_price) / current_price) * 100
                    
                    self.print_color(f"{direction_icon} {pair} {trade['direction']}", self.Fore.WHITE + self.Style.BRIGHT)
                    self.print_color(f"   📊 Quantity: {trade['quantity']} | ⚡ {leverage}x", self.Fore.WHITE)
                    self.print_color(f"   💰 Entry: ${entry_price:.4f} | Current: ${current_price:.4f}", self.Fore.CYAN)
                    self.print_color(f"   💸 P&L: ${live_data['unrealized_pnl']:.2f} ({pnl_percent:+.2f}%)", pnl_color)
                    self.print_color(f"   🎯 TP: ${trade['take_profit']:.4f} ({tp_distance:.2f}% to go)", self.Fore.GREEN)
                    self.print_color(f"   🛡️  SL: ${trade['stop_loss']:.4f} ({sl_distance:.2f}% to go)", self.Fore.RED)
                    self.print_color(f"   🔥 AI Confidence: {trade['ai_confidence']}%", self.Fore.MAGENTA)
                    self.print_color("   " + "-" * 80, self.Fore.MAGENTA)
            
        except Exception as e:
            self.print_color(f"❌ Error getting bot positions: {e}", self.Fore.RED)

    def show_trade_history_menu(self):
        """Trade history ကိုသပ်သပ်ကြည့်ရန် menu"""
        while True:
            try:
                print("\n" + "="*80)
                print("📋 TRADE HISTORY MENU")
                print("="*80)
                print("1. Show Last 10 Trades")
                print("2. Show Last 20 Trades") 
                print("3. Show All Trades")
                print("4. Show Profit/Loss Summary")
                print("5. Back to Main")
                
                choice = input("Select option (1-5): ").strip()
                
                if choice == "1":
                    self.show_trade_history(10)
                elif choice == "2":
                    self.show_trade_history(20)
                elif choice == "3":
                    self.show_trade_history(100)
                elif choice == "4":
                    self.show_pnl_summary()
                elif choice == "5":
                    break
                else:
                    print("Invalid choice!")
                    
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(2)

    def show_pnl_summary(self):
        """Overall P&L summary ပြရန်"""
        if not self.trade_history:
            self.print_color("No trade history found", self.Fore.YELLOW)
            return
        
        total_trades = len(self.trade_history)
        winning_trades = len([t for t in self.trade_history if t.get('pnl', 0) > 0])
        losing_trades = len([t for t in self.trade_history if t.get('pnl', 0) < 0])
        total_pnl = sum(trade.get('pnl', 0) for trade in self.trade_history)
        avg_win = np.mean([t.get('pnl', 0) for t in self.trade_history if t.get('pnl', 0) > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t.get('pnl', 0) for t in self.trade_history if t.get('pnl', 0) < 0]) if losing_trades > 0 else 0
        
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        self.print_color(f"\n💰 P&L SUMMARY", self.Fore.CYAN + self.Style.BRIGHT)
        self.print_color("=" * 80, self.Fore.CYAN)
        self.print_color(f"📊 Total Trades: {total_trades}", self.Fore.WHITE)
        self.print_color(f"✅ Winning Trades: {winning_trades}", self.Fore.GREEN)
        self.print_color(f"❌ Losing Trades: {losing_trades}", self.Fore.RED)
        self.print_color(f"🎯 Win Rate: {win_rate:.1f}%", self.Fore.GREEN if win_rate > 50 else self.Fore.YELLOW)
        self.print_color(f"📈 Total P&L: ${total_pnl:.2f}", self.Fore.GREEN if total_pnl > 0 else self.Fore.RED)
        self.print_color(f"📊 Average Win: ${avg_win:.2f}", self.Fore.GREEN)
        self.print_color(f"📊 Average Loss: ${avg_loss:.2f}", self.Fore.RED)
        if avg_loss != 0:
            profit_factor = abs(avg_win / avg_loss)
            self.print_color(f"📈 Profit Factor: {profit_factor:.2f}", self.Fore.CYAN)

    def get_live_position_data(self, pair):
        """လက်ရှိ position data ကိုရယူရန်"""
        try:
            positions = self.binance.futures_position_information(symbol=pair)
            for pos in positions:
                if pos['symbol'] == pair and float(pos['positionAmt']) != 0:
                    entry_price = float(pos.get('entryPrice', 0))
                    quantity = abs(float(pos['positionAmt']))
                    unrealized_pnl = float(pos.get('unRealizedProfit', 0))
                    
                    ticker = self.binance.futures_symbol_ticker(symbol=pair)
                    current_price = float(ticker['price'])
                    
                    direction = "SHORT" if float(pos['positionAmt']) < 0 else "LONG"
                    
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
            self.print_color(f"❌ Error getting live data for {pair}: {e}", self.Fore.RED)
            return None

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
            if len(self.trade_history) > 100:
                self.trade_history = self.trade_history[-100:]
            self.save_trade_history()
        except Exception as e:
            self.print_color(f"Error adding trade to history: {e}", self.Fore.RED)
    
    def show_trade_history(self, limit=10):
        if not self.trade_history:
            self.print_color("No trade history found", self.Fore.YELLOW)
            return
        
        self.print_color(f"\n📋 TRADE HISTORY (Last {min(limit, len(self.trade_history))} trades)", self.Fore.CYAN + self.Style.BRIGHT)
        self.print_color("=" * 120, self.Fore.CYAN)
        
        for i, trade in enumerate(reversed(self.trade_history[-limit:])):
            pnl = trade.get('pnl', 0)
            pnl_color = self.Fore.GREEN if pnl > 0 else self.Fore.RED if pnl < 0 else self.Fore.YELLOW
            direction_icon = "🟢 LONG" if trade['direction'] == 'LONG' else "🔴 SHORT"
            
            # Calculate percentages
            entry_price = trade.get('entry_price', 1)
            exit_price = trade.get('exit_price', 1)
            tp_price = trade.get('take_profit', 1)
            sl_price = trade.get('stop_loss', 1)
            
            if entry_price > 0:
                pnl_percent = (pnl / (entry_price * trade.get('quantity', 1))) * 100 * 100
                tp_percent = ((tp_price - entry_price) / entry_price) * 100
                sl_percent = ((sl_price - entry_price) / entry_price) * 100
                
                if trade['direction'] == 'SHORT':
                    tp_percent = -tp_percent
                    sl_percent = -sl_percent
            else:
                pnl_percent = 0
                tp_percent = 0
                sl_percent = 0
            
            self.print_color(f"{i+1}. {direction_icon} {trade['pair']} | P&L: ${pnl:.2f} ({pnl_percent:+.2f}%)", pnl_color)
            self.print_color(f"   📊 Entry: ${entry_price:.4f} | Exit: ${exit_price:.4f}", self.Fore.WHITE)
            self.print_color(f"   🎯 TP: ${tp_price:.4f} ({tp_percent:+.2f}%) | 🛡️ SL: ${sl_price:.4f} ({sl_percent:+.2f}%)", self.Fore.CYAN)
            self.print_color(f"   ⏰ Time: {trade.get('close_time', 'N/A')}", self.Fore.YELLOW)
            if trade.get('close_reason'):
                self.print_color(f"   📝 Reason: {trade['close_reason']}", self.Fore.WHITE)
            self.print_color("   " + "-" * 100, self.Fore.CYAN)
    
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
        if not all([self.binance_api_key, self.binance_secret, self.deepseek_key]):
            self.print_color("Missing API keys!", self.Fore.RED)
            return False
        try:
            if self.binance:
                self.binance.futures_exchange_info()
                self.print_color("✅ Binance connection successful!", self.Fore.GREEN)
            else:
                self.print_color("Binance client not available", self.Fore.YELLOW)
                return False
        except Exception as e:
            self.print_color(f"❌ Binance connection failed: {e}", self.Fore.RED)
            return False
        return True

    def setup_futures(self):
        if not self.binance:
            return
        try:
            for pair in self.available_pairs:
                try:
                    self.binance.futures_change_leverage(symbol=pair, leverage=self.leverage)
                    self.print_color(f"✅ Leverage set for {pair}", self.Fore.GREEN)
                except Exception as e:
                    self.print_color(f"⚠️  Leverage setup failed for {pair}: {e}", self.Fore.YELLOW)
            self.print_color("✅ Futures setup completed!", self.Fore.GREEN)
        except Exception as e:
            self.print_color(f"❌ Futures setup failed: {e}", self.Fore.RED)
    
    def load_symbol_precision(self):
        if not self.binance:
            for pair in self.available_pairs:
                self.quantity_precision[pair] = 3
                self.price_precision[pair] = 4
            self.print_color("📊 Default precision set", self.Fore.GREEN)
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
            self.print_color(f"❌ Error loading symbol precision: {e}", self.Fore.RED)
    
    def format_price(self, pair, price):
        if price <= 0:
            return 0.0
        precision = self.price_precision.get(pair, 4)
        return round(price, precision)
    
    def get_quantity(self, pair, price):
        try:
            if not price or price <= 0:
                self.print_color(f"❌ Invalid price: {price} for {pair}", self.Fore.RED)
                return None
            
            # Fixed $50 position size calculation
            quantity = self.trade_size_usd * self.leverage / price
            quantity = max(quantity, 0.001)
            
            precision = self.quantity_precision.get(pair, 3)
            quantity = round(quantity, precision)
            
            if quantity <= 0:
                self.print_color(f"❌ Invalid quantity: {quantity} for {pair}", self.Fore.RED)
                return None
            
            actual_value = quantity * price
            self.print_color(f"📊 Quantity for {pair}: {quantity} = ${actual_value:.2f} (${self.trade_size_usd} × {self.leverage}x)", self.Fore.CYAN)
            return quantity
        except Exception as e:
            self.print_color(f"❌ Quantity calculation failed: {e}", self.Fore.RED)
            return None

    def parse_ai_response(self, text):
        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                decision_data = json.loads(json_str)
                
                direction = decision_data.get('direction', 'HOLD').upper()
                entry_price = float(decision_data.get('entry_price', 0))
                take_profit_percent = float(decision_data.get('take_profit_percent', 1.5))  # Higher TP for aggressive
                stop_loss_percent = float(decision_data.get('stop_loss_percent', 0.8))     # Tighter SL for aggressive
                confidence = float(decision_data.get('confidence', 50))
                reason = decision_data.get('reason', 'AI Analysis')
                
                # Validate inputs
                if direction not in ['LONG', 'SHORT', 'HOLD']:
                    direction = 'HOLD'
                if confidence < 0 or confidence > 100:
                    confidence = 50
                if entry_price <= 0:
                    entry_price = None
                if take_profit_percent <= 0:
                    take_profit_percent = 1.5
                if stop_loss_percent <= 0:
                    stop_loss_percent = 0.8
                
                return direction, entry_price, take_profit_percent, stop_loss_percent, confidence, reason
            return 'HOLD', None, 1.5, 0.8, 50, 'No valid JSON found'
        except Exception as e:
            self.print_color(f"❌ AI response parsing failed: {e}", self.Fore.RED)
            return 'HOLD', None, 1.5, 0.8, 50, 'Parsing failed'

    def get_deepseek_analysis(self, pair, market_data):
        try:
            if not self.deepseek_key:
                self.print_color("❌ DeepSeek API key not found", self.Fore.RED)
                return "HOLD", None, 1.5, 0.8, 0, "No API key"
            
            current_price = market_data['current_price']
            price_change = market_data.get('price_change_percent', 0)
            
            prompt = f"""
            Analyze {pair} for AGGRESSIVE 3-minute scalping with EXACT parameters.
            
            MARKET DATA:
            - Current: ${current_price:.6f}
            - Change: {price_change:.2f}%
            - Recent: {market_data.get('prices', [])[-5:]}
            
            AGGRESSIVE TRADING RULES:
            - Be MORE AGGRESSIVE with entries
            - Take profit 1.0-3.0% 
            - Stop loss 0.5-1.5%
            - Look for quick momentum moves
            - Higher risk for higher rewards
            
            DECIDE ALL PARAMETERS:
            - LONG/SHORT/HOLD
            - Exact entry price
            - Take profit % (1.0-3.0%)
            - Stop loss % (0.5-1.5%) 
            - Confidence (0-100)
            - Brief reason
            
            Return VALID JSON only:
            {{
                "direction": "LONG" | "SHORT" | "HOLD",
                "entry_price": number,
                "take_profit_percent": number,
                "stop_loss_percent": number, 
                "confidence": number,
                "reason": "string"
            }}
            """
            
            headers = {
                "Authorization": f"Bearer {self.deepseek_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are an AGGRESSIVE 3-min scalper. Take more risks for higher rewards. Return perfect JSON with higher TP and tighter SL. You decide everything - direction, entry, TP%, SL%."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.6,  # Higher temperature for more aggressive decisions
                "max_tokens": 400
            }
            
            self.print_color(f"🧠 AGGRESSIVE AI Analyzing {pair}...", self.Fore.MAGENTA)
            response = requests.post("https://api.deepseek.com/chat/completions", 
                                   headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['choices'][0]['message']['content'].strip()
                direction, entry_price, tp_percent, sl_percent, confidence, reason = self.parse_ai_response(ai_response)
                
                direction_icon = "🟢 LONG" if direction == "LONG" else "🔴 SHORT" if direction == "SHORT" else "🟡 HOLD"
                color = self.Fore.BLUE if direction == "LONG" else self.Fore.RED if direction == "SHORT" else self.Fore.YELLOW
                
                if direction != "HOLD":
                    self.print_color(f"{direction_icon} {direction} @ ${entry_price:.4f}", color + self.Style.BRIGHT)
                    self.print_color(f"   🎯 TP: +{tp_percent:.2f}% | 🛡️ SL: -{sl_percent:.2f}% | 🔥 Confidence: {confidence}%", color)
                    self.print_color(f"   📝 Reason: {reason}", self.Fore.WHITE)
                else:
                    self.print_color(f"{direction_icon} Confidence: {confidence}% - {reason}", color)
                
                return direction, entry_price, tp_percent, sl_percent, confidence, reason
            else:
                self.print_color(f"❌ DeepSeek API error: {response.status_code}", self.Fore.RED)
                return "HOLD", None, 1.5, 0.8, 0, f"API Error"
                
        except Exception as e:
            self.print_color(f"❌ DeepSeek analysis failed: {e}", self.Fore.RED)
            return "HOLD", None, 1.5, 0.8, 0, f"Error"

    def get_price_history(self, pair, limit=10):  # Shorter history for faster decisions
        try:
            if self.binance:
                klines = self.binance.futures_klines(
                    symbol=pair, 
                    interval=Client.KLINE_INTERVAL_3MINUTE, 
                    limit=limit
                )
                prices = [float(k[4]) for k in klines]  # Close prices
                current_price = prices[-1] if prices else 0
                
                # Calculate price change percentage
                if len(prices) >= 2:
                    price_change = ((current_price - prices[-2]) / prices[-2]) * 100
                else:
                    price_change = 0
                
                return {
                    'prices': prices,
                    'current_price': current_price,
                    'price_change_percent': price_change
                }
            else:
                current_price = self.get_current_price(pair)
                return {
                    'prices': [current_price] * 10,
                    'current_price': current_price,
                    'price_change_percent': 0
                }
        except Exception as e:
            current_price = self.get_current_price(pair)
            return {
                'prices': [current_price] * 10,
                'current_price': current_price,
                'price_change_percent': 0
            }

    def get_ai_decision(self, pair_data):
        try:
            pair = list(pair_data.keys())[0]
            current_price = pair_data[pair]['price']
            
            if current_price <= 0:
                return {
                    "action": "HOLD", 
                    "pair": pair, 
                    "direction": "HOLD", 
                    "confidence": 0, 
                    "reason": "Invalid price"
                }
            
            self.print_color(f"📈 Aggressive Analysis {pair} at ${current_price:.4f}...", self.Fore.BLUE)
            market_data = self.get_price_history(pair)
            market_data['current_price'] = current_price
            
            direction, entry_price, tp_percent, sl_percent, confidence, reason = self.get_deepseek_analysis(pair, market_data)
            
            if direction == "HOLD" or confidence < self.confidence_threshold:
                self.print_color(f"🟡 HOLD ({confidence}% confidence)", self.Fore.YELLOW)
                return {
                    "action": "HOLD", 
                    "pair": pair, 
                    "direction": direction, 
                    "confidence": confidence, 
                    "reason": reason
                }
            else:
                direction_icon = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"
                color = self.Fore.BLUE if direction == "LONG" else self.Fore.RED
                
                self.print_color(f"🚀 AGGRESSIVE SIGNAL: {direction_icon}", color + self.Style.BRIGHT)
                self.print_color(f"   💰 Entry: ${entry_price:.4f} | 🎯 TP: +{tp_percent:.2f}% | 🛡️ SL: -{sl_percent:.2f}%", color)
                self.print_color(f"   🔥 Confidence: {confidence}% | 📝 {reason}", color)
                
                return {
                    "action": "TRADE",
                    "pair": pair,
                    "direction": direction,
                    "entry_price": entry_price,
                    "take_profit_percent": tp_percent,
                    "stop_loss_percent": sl_percent,
                    "confidence": confidence,
                    "reason": reason
                }
                
        except Exception as e:
            self.print_color(f"❌ AI decision failed: {e}", self.Fore.RED)
            return {
                "action": "HOLD", 
                "pair": list(pair_data.keys())[0], 
                "direction": "HOLD", 
                "confidence": 0, 
                "reason": f"Error: {str(e)}"
            }

    def get_current_price(self, pair):
        try:
            if self.binance:
                ticker = self.binance.futures_symbol_ticker(symbol=pair)
                return float(ticker['price'])
            else:
                # Fallback for paper trading
                base_prices = {
                    "SOLUSDT": 180.50, "AVAXUSDT": 35.20, "XRPUSDT": 0.62,
                    "LINKUSDT": 18.75, "DOTUSDT": 8.90, "ADAUSDT": 0.48,
                    "MATICUSDT": 0.85
                }
                return base_prices.get(pair, 100)
        except:
            return None

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
        return len([t for t in self.bot_opened_trades.values() if t['status'] == 'ACTIVE']) < self.max_concurrent_trades

    def execute_trade(self, decision):
        try:
            pair = decision["pair"]
            
            if not self.can_open_new_trade(pair):
                self.print_color(f"⏸️  Cannot open {pair} - position exists or max trades reached", self.Fore.YELLOW)
                return False
            
            direction = decision["direction"]
            entry_price = decision["entry_price"]
            tp_percent = decision["take_profit_percent"]
            sl_percent = decision["stop_loss_percent"]
            confidence = decision["confidence"]
            reason = decision["reason"]
            
            if entry_price is None or entry_price <= 0:
                self.print_color(f"❌ Invalid AI entry price", self.Fore.RED)
                return False
            
            quantity = self.get_quantity(pair, entry_price)
            if quantity is None:
                return False
            
            # Calculate TP/SL prices based on AI percentages
            if direction == "LONG":
                take_profit = entry_price * (1 + tp_percent / 100)
                stop_loss = entry_price * (1 - sl_percent / 100)
            else:  # SHORT
                take_profit = entry_price * (1 - tp_percent / 100)
                stop_loss = entry_price * (1 + sl_percent / 100)
            
            take_profit = self.format_price(pair, take_profit)
            stop_loss = self.format_price(pair, stop_loss)
            
            direction_color = self.Fore.BLUE if direction == 'LONG' else self.Fore.RED
            direction_icon = "🟢 LONG" if direction == 'LONG' else "🔴 SHORT"
            
            # Display trade execution details
            self.print_color(f"\n🎯 AGGRESSIVE TRADE EXECUTION", self.Fore.CYAN + self.Style.BRIGHT)
            self.print_color("=" * 70, self.Fore.CYAN)
            self.print_color(f"{direction_icon} {pair}", direction_color + self.Style.BRIGHT)
            self.print_color(f"💰 Entry: ${entry_price:.4f}", self.Fore.GREEN)
            self.print_color(f"📊 Quantity: {quantity} (${self.trade_size_usd} × {self.leverage}x)", self.Fore.WHITE)
            self.print_color(f"🎯 Take Profit: ${take_profit:.4f} (+{tp_percent:.2f}%)", self.Fore.GREEN)
            self.print_color(f"🛡️ Stop Loss: ${stop_loss:.4f} (-{sl_percent:.2f}%)", self.Fore.RED)
            self.print_color(f"🔥 AI Confidence: {confidence}%", self.Fore.MAGENTA)
            self.print_color(f"📝 Reason: {reason}", self.Fore.YELLOW)
            self.print_color("=" * 70, self.Fore.CYAN)
            
            # Execute trade
            entry_side = 'BUY' if direction == 'LONG' else 'SELL'
            
            try:
                # Place entry order
                order = self.binance.futures_create_order(
                    symbol=pair,
                    side=entry_side,
                    type='MARKET',
                    quantity=quantity
                )
                
                self.print_color(f"✅ {direction} ORDER EXECUTED!", self.Fore.GREEN + self.Style.BRIGHT)
                time.sleep(1)  # Small delay for order execution
                
                # Place stop loss and take profit orders
                stop_side = 'SELL' if direction == 'LONG' else 'BUY'
                
                # Stop Loss
                self.binance.futures_create_order(
                    symbol=pair, 
                    side=stop_side, 
                    type='STOP_MARKET',
                    quantity=quantity, 
                    stopPrice=stop_loss, 
                    reduceOnly=True
                )
                
                # Take Profit  
                self.binance.futures_create_order(
                    symbol=pair, 
                    side=stop_side, 
                    type='TAKE_PROFIT_MARKET',
                    quantity=quantity, 
                    stopPrice=take_profit, 
                    reduceOnly=True
                )
                
                # Track the trade
                self.bot_opened_trades[pair] = {
                    "pair": pair, 
                    "direction": direction, 
                    "entry_price": entry_price,
                    "quantity": quantity, 
                    "stop_loss": stop_loss, 
                    "take_profit": take_profit,
                    "take_profit_percent": tp_percent,
                    "stop_loss_percent": sl_percent,
                    "entry_time": time.time(), 
                    "status": 'ACTIVE', 
                    'ai_confidence': confidence,
                    'ai_reason': reason, 
                    'entry_time_th': self.get_thailand_time()
                }
                
                self.print_color(f"✅ AGGRESSIVE TRADE ACTIVATED: {pair} {direction}", self.Fore.GREEN + self.Style.BRIGHT)
                
                # Immediately show updated positions after trade
                time.sleep(2)
                self.show_all_positions_dashboard()
                self.show_bot_managed_positions()
                
                return True
                
            except Exception as e:
                self.print_color(f"❌ Execution Error: {e}", self.Fore.RED)
                return False
                
        except Exception as e:
            self.print_color(f"❌ Trade execution failed: {e}", self.Fore.RED)
            return False

    def monitor_positions(self):
        try:
            for pair, trade in list(self.bot_opened_trades.items()):
                if trade['status'] != 'ACTIVE':
                    continue
                    
                live_data = self.get_live_position_data(pair)
                if not live_data:
                    self.close_trade_with_cleanup(pair, trade)
                    continue
                
                # Check if position hit TP/SL
                current_price = live_data['current_price']
                should_close = False
                close_reason = ""
                
                if trade['direction'] == 'LONG':
                    if current_price >= trade['take_profit']:
                        should_close = True
                        close_reason = "TP HIT"
                    elif current_price <= trade['stop_loss']:
                        should_close = True
                        close_reason = "SL HIT"
                else:  # SHORT
                    if current_price <= trade['take_profit']:
                        should_close = True
                        close_reason = "TP HIT"
                    elif current_price >= trade['stop_loss']:
                        should_close = True
                        close_reason = "SL HIT"
                
                if should_close:
                    self.print_color(f"🔚 Position closed: {pair} - {close_reason}", self.Fore.CYAN)
                    self.close_trade_with_cleanup(pair, trade)
                    
        except Exception as e:
            self.print_color(f"❌ Monitoring error: {e}", self.Fore.RED)

    def close_trade_with_cleanup(self, pair, trade):
        try:
            # Cancel any open orders
            open_orders = self.binance.futures_get_open_orders(symbol=pair)
            canceled = 0
            for order in open_orders:
                if order['reduceOnly'] and order['symbol'] == pair:
                    try:
                        self.binance.futures_cancel_order(symbol=pair, orderId=order['orderId'])
                        canceled += 1
                    except: 
                        pass
            
            # Calculate final P&L
            final_pnl = self.get_final_pnl(pair, trade)
            
            # Update trade status
            trade['status'] = 'CLOSED'
            trade['exit_time_th'] = self.get_thailand_time()
            trade['exit_price'] = self.get_current_price(pair)
            trade['pnl'] = final_pnl
            trade['close_reason'] = "Auto Closed"
            
            # Add to history
            closed_trade = trade.copy()
            self.add_trade_to_history(closed_trade)
            
            # Display closure info
            pnl_color = self.Fore.GREEN if final_pnl > 0 else self.Fore.RED
            direction_icon = "🟢 LONG" if trade['direction'] == 'LONG' else "🔴 SHORT"
            
            self.print_color(f"\n🔚 TRADE CLOSED: {pair} {direction_icon}", pnl_color + self.Style.BRIGHT)
            self.print_color(f"   💰 Final P&L: ${final_pnl:.2f}", pnl_color)
            if canceled > 0:
                self.print_color(f"   🧹 Cleaned up {canceled} order(s)", self.Fore.CYAN)
                
            # Remove from active trades
            if pair in self.bot_opened_trades:
                del self.bot_opened_trades[pair]
                
        except Exception as e:
            self.print_color(f"❌ Cleanup failed for {pair}: {e}", self.Fore.RED)

    def get_final_pnl(self, pair, trade):
        try:
            live_data = self.get_live_position_data(pair)
            if live_data and 'unrealized_pnl' in live_data:
                return live_data['unrealized_pnl']
            
            current_price = self.get_current_price(pair)
            if not current_price:
                return 0
                
            if trade['direction'] == 'LONG':
                return (current_price - trade['entry_price']) * trade['quantity']
            else:
                return (trade['entry_price'] - current_price) * trade['quantity']
        except:
            return 0

    def run_trading_cycle(self):
        try:
            self.monitor_positions()
            
            # Show live positions every cycle
            self.show_all_positions_dashboard()
            self.show_bot_managed_positions()
            
            # Show history every 10 cycles
            if hasattr(self, 'cycle_count') and self.cycle_count % 10 == 0:
                self.show_trade_history(5)
            
            market_data = self.get_market_data()
            if market_data:
                self.print_color(f"\n🔍 AGGRESSIVE AI SCANNING {len(market_data)} PAIRS...", self.Fore.BLUE + self.Style.BRIGHT)
                
                for pair in market_data.keys():
                    if self.can_open_new_trade(pair):
                        pair_data = {pair: market_data[pair]}
                        decision = self.get_ai_decision(pair_data)
                        
                        if decision["action"] == "TRADE":
                            direction_icon = "🟢 LONG" if decision['direction'] == "LONG" else "🔴 SHORT"
                            self.print_color(f"🚀 EXECUTING AGGRESSIVE TRADE: {pair} {direction_icon}", self.Fore.GREEN + self.Style.BRIGHT)
                            success = self.execute_trade(decision)
                        else:
                            self.print_color(f"🟡 HOLD: {pair} ({decision['confidence']}%)", self.Fore.YELLOW)
                    else:
                        self.print_color(f"⏸️  SKIPPED: {pair} (position exists or max trades)", self.Fore.MAGENTA)
            else:
                self.print_color("📭 No market data available", self.Fore.YELLOW)
                
        except Exception as e:
            self.print_color(f"❌ Trading cycle error: {e}", self.Fore.RED)

    def start_trading(self):
        self.print_color("\n🎯 STARTING AGGRESSIVE AI TRADING BOT!", self.Fore.RED + self.Style.BRIGHT)
        self.print_color("⚠️  REAL MONEY TRADING - AGGRESSIVE MODE!", self.Fore.RED + self.Style.BRIGHT)
        self.print_color(f"💰 Fixed Size: ${self.trade_size_usd} | ⚡ Leverage: {self.leverage}x", self.Fore.YELLOW)
        self.print_color(f"🔥 Confidence Threshold: {self.confidence_threshold}% | Max Trades: {self.max_concurrent_trades}", self.Fore.MAGENTA)
        
        # Show current positions on startup
        self.show_all_positions_dashboard()
        self.show_bot_managed_positions()
        
        self.cycle_count = 0
        while True:
            try:
                self.cycle_count += 1
                self.print_color(f"\n🔄 AGGRESSIVE TRADING CYCLE {self.cycle_count}", self.Fore.CYAN + self.Style.BRIGHT)
                self.print_color("=" * 50, self.Fore.CYAN)
                
                self.run_trading_cycle()
                
                self.print_color(f"⏰ Waiting 20 seconds for next aggressive analysis...", self.Fore.BLUE)
                time.sleep(20)  # Shorter wait for aggressive mode
                
            except KeyboardInterrupt:
                self.print_color(f"\n🛑 AGGRESSIVE TRADING STOPPED BY USER", self.Fore.RED + self.Style.BRIGHT)
                
                # Control C နဲ့ရပ်လိုက်ရင်လည်း positions တွေဆက်ကြည့်လို့ရအောင်
                while True:
                    try:
                        print("\n" + "="*80)
                        print("🛑 TRADING STOPPED - OPTIONS")
                        print("="*80)
                        print("1. 📊 View Current Positions")
                        print("2. 📋 View Trade History") 
                        print("3. 🔄 Restart Trading")
                        print("4. 🚪 Exit to Main Menu")
                        
                        choice = input("Select option (1-4): ").strip()
                        
                        if choice == "1":
                            self.show_all_positions_dashboard()
                            self.show_bot_managed_positions()
                            input("\nPress Enter to continue...")
                        elif choice == "2":
                            self.show_trade_history_menu()
                        elif choice == "3":
                            self.start_trading()  # Restart trading
                            break
                        elif choice == "4":
                            break
                        else:
                            print("Invalid choice!")
                            
                    except KeyboardInterrupt:
                        print("\nReturning to main menu...")
                        break
                    except Exception as e:
                        print(f"Error: {e}")
                        time.sleep(2)
                break
                
            except Exception as e:
                self.print_color(f"❌ Main loop error: {e}", self.Fore.RED)
                time.sleep(20)


class AggressiveAIPaperTradingBot:
    def __init__(self, real_bot):
        self.real_bot = real_bot
        self.Fore = real_bot.Fore
        self.Back = real_bot.Back  
        self.Style = real_bot.Style
        self.COLORAMA_AVAILABLE = real_bot.COLORAMA_AVAILABLE
        
        self.paper_balance = 1000
        self.paper_positions = {}
        self.paper_history = []
        
        self.real_bot.print_color("🤖 AGGRESSIVE AI PAPER TRADING BOT INITIALIZED!", self.Fore.GREEN + self.Style.BRIGHT)
        self.real_bot.print_color(f"💰 Starting Paper Balance: ${self.paper_balance}", self.Fore.CYAN)
        self.real_bot.print_color(f"🎯 AGGRESSIVE AI Controls Everything | Fixed Size: ${real_bot.trade_size_usd}", self.Fore.MAGENTA)

    def monitor_paper_positions(self):
        try:
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
                        close_reason = "🎯 TP HIT"
                        pnl = (current_price - trade['entry_price']) * trade['quantity']
                    elif current_price <= trade['stop_loss']:
                        should_close = True
                        close_reason = "🛡️ SL HIT" 
                        pnl = (current_price - trade['entry_price']) * trade['quantity']
                else:  # SHORT
                    if current_price <= trade['take_profit']:
                        should_close = True
                        close_reason = "🎯 TP HIT"
                        pnl = (trade['entry_price'] - current_price) * trade['quantity']
                    elif current_price >= trade['stop_loss']:
                        should_close = True
                        close_reason = "🛡️ SL HIT"
                        pnl = (trade['entry_price'] - current_price) * trade['quantity']
                
                if should_close:
                    trade['status'] = 'CLOSED'
                    trade['exit_price'] = current_price
                    trade['pnl'] = pnl
                    trade['close_reason'] = close_reason
                    trade['close_time'] = self.real_bot.get_thailand_time()
                    
                    self.paper_balance += pnl
                    self.paper_history.append(trade.copy())
                    
                    pnl_color = self.Fore.GREEN if pnl > 0 else self.Fore.RED
                    direction_icon = "🟢 LONG" if trade['direction'] == 'LONG' else "🔴 SHORT"
                    
                    self.real_bot.print_color(f"\n🔚 PAPER TRADE CLOSED: {pair} {direction_icon}", pnl_color)
                    self.real_bot.print_color(f"   💰 P&L: ${pnl:.2f} | Reason: {close_reason}", pnl_color)
                    
                    del self.paper_positions[pair]
                    
        except Exception as e:
            self.real_bot.print_color(f"❌ Paper monitoring error: {e}", self.Fore.RED)

    def paper_execute_trade(self, decision):
        try:
            pair = decision["pair"]
            direction = decision["direction"]
            entry_price = decision["entry_price"]
            tp_percent = decision["take_profit_percent"]
            sl_percent = decision["stop_loss_percent"]
            confidence = decision["confidence"]
            reason = decision["reason"]
            
            if entry_price is None or entry_price <= 0:
                return False
                
            quantity = self.real_bot.get_quantity(pair, entry_price)
            if quantity is None:
                return False
            
            # Calculate TP/SL prices based on AI percentages
            if direction == "LONG":
                take_profit = entry_price * (1 + tp_percent / 100)
                stop_loss = entry_price * (1 - sl_percent / 100)
            else:  # SHORT
                take_profit = entry_price * (1 - tp_percent / 100)
                stop_loss = entry_price * (1 + sl_percent / 100)
            
            take_profit = self.real_bot.format_price(pair, take_profit)
            stop_loss = self.real_bot.format_price(pair, stop_loss)
            
            direction_color = self.Fore.BLUE if direction == 'LONG' else self.Fore.RED
            direction_icon = "🟢 LONG" if direction == 'LONG' else "🔴 SHORT"
            
            self.real_bot.print_color(f"\n📝 PAPER TRADE EXECUTION", self.Fore.CYAN + self.Style.BRIGHT)
            self.real_bot.print_color("=" * 70, self.Fore.CYAN)
            self.real_bot.print_color(f"{direction_icon} {pair}", direction_color + self.Style.BRIGHT)
            self.real_bot.print_color(f"💰 Entry: ${entry_price:.4f}", self.Fore.GREEN)
            self.real_bot.print_color(f"📊 Quantity: {quantity}", self.Fore.WHITE)
            self.real_bot.print_color(f"🎯 Take Profit: ${take_profit:.4f} (+{tp_percent:.2f}%)", self.Fore.GREEN)
            self.real_bot.print_color(f"🛡️ Stop Loss: ${stop_loss:.4f} (-{sl_percent:.2f}%)", self.Fore.RED)
            self.real_bot.print_color(f"🔥 AI Confidence: {confidence}%", self.Fore.MAGENTA)
            self.real_bot.print_color(f"📝 Reason: {reason}", self.Fore.YELLOW)
            self.real_bot.print_color("=" * 70, self.Fore.CYAN)
            
            self.paper_positions[pair] = {
                "pair": pair, 
                "direction": direction, 
                "entry_price": entry_price,
                "quantity": quantity, 
                "stop_loss": stop_loss, 
                "take_profit": take_profit,
                "take_profit_percent": tp_percent,
                "stop_loss_percent": sl_percent,
                "entry_time": time.time(), 
                "status": 'ACTIVE', 
                'ai_confidence': confidence,
                'ai_reason': reason, 
                'entry_time_th': self.real_bot.get_thailand_time()
            }
            return True
            
        except Exception as e:
            self.real_bot.print_color(f"❌ Paper trade failed: {e}", self.Fore.RED)
            return False

    def get_paper_portfolio_status(self):
        total_trades = len(self.paper_history)
        winning_trades = len([t for t in self.paper_history if t.get('pnl', 0) > 0])
        total_pnl = sum(trade.get('pnl', 0) for trade in self.paper_history)
        
        self.real_bot.print_color(f"\n📊 PAPER TRADING PORTFOLIO", self.Fore.CYAN + self.Style.BRIGHT)
        self.real_bot.print_color("=" * 60, self.Fore.CYAN)
        self.real_bot.print_color(f"📈 Active Positions: {len(self.paper_positions)}", self.Fore.WHITE)
        self.real_bot.print_color(f"💰 Balance: ${self.paper_balance:.2f}", self.Fore.WHITE)
        self.real_bot.print_color(f"📋 Total Trades: {total_trades}", self.Fore.WHITE)
        
        if total_trades > 0:
            win_rate = (winning_trades / total_trades) * 100
            self.real_bot.print_color(f"🎯 Win Rate: {win_rate:.1f}%", 
                                    self.Fore.GREEN if win_rate > 50 else self.Fore.YELLOW)
            self.real_bot.print_color(f"📊 Total P&L: ${total_pnl:.2f}", 
                                    self.Fore.GREEN if total_pnl > 0 else self.Fore.RED)

    def show_paper_positions(self):
        """Show current paper trading positions"""
        if not self.paper_positions:
            self.real_bot.print_color("📭 No active paper positions", self.Fore.YELLOW)
            return
        
        self.real_bot.print_color(f"\n📝 PAPER TRADING POSITIONS", self.Fore.MAGENTA + self.Style.BRIGHT)
        self.real_bot.print_color("=" * 100, self.Fore.MAGENTA)
        
        for pair, trade in self.paper_positions.items():
            if trade['status'] == 'ACTIVE':
                current_price = self.real_bot.get_current_price(pair)
                if current_price:
                    direction_icon = "🟢" if trade['direction'] == 'LONG' else "🔴"
                    
                    if trade['direction'] == 'LONG':
                        pnl = (current_price - trade['entry_price']) * trade['quantity']
                        pnl_percent = ((current_price - trade['entry_price']) / trade['entry_price']) * 100 * self.real_bot.leverage
                        tp_distance = ((trade['take_profit'] - current_price) / current_price) * 100
                        sl_distance = ((current_price - trade['stop_loss']) / current_price) * 100
                    else:
                        pnl = (trade['entry_price'] - current_price) * trade['quantity']
                        pnl_percent = ((trade['entry_price'] - current_price) / trade['entry_price']) * 100 * self.real_bot.leverage
                        tp_distance = ((current_price - trade['take_profit']) / current_price) * 100
                        sl_distance = ((trade['stop_loss'] - current_price) / current_price) * 100
                    
                    pnl_color = self.Fore.GREEN if pnl >= 0 else self.Fore.RED
                    
                    self.real_bot.print_color(f"{direction_icon} {pair} {trade['direction']}", self.Fore.WHITE + self.Style.BRIGHT)
                    self.real_bot.print_color(f"   📊 Quantity: {trade['quantity']} | ⚡ {self.real_bot.leverage}x", self.Fore.WHITE)
                    self.real_bot.print_color(f"   💰 Entry: ${trade['entry_price']:.4f} | Current: ${current_price:.4f}", self.Fore.CYAN)
                    self.real_bot.print_color(f"   💸 P&L: ${pnl:.2f} ({pnl_percent:+.2f}%)", pnl_color)
                    self.real_bot.print_color(f"   🎯 TP: ${trade['take_profit']:.4f} ({tp_distance:.2f}% to go)", self.Fore.GREEN)
                    self.real_bot.print_color(f"   🛡️  SL: ${trade['stop_loss']:.4f} ({sl_distance:.2f}% to go)", self.Fore.RED)
                    self.real_bot.print_color("   " + "-" * 80, self.Fore.MAGENTA)

    def run_paper_trading_cycle(self):
        try:
            self.monitor_paper_positions()
            market_data = self.real_bot.get_market_data()
            
            # Show paper positions every cycle
            self.show_paper_positions()
            self.get_paper_portfolio_status()
            
            if market_data:
                self.real_bot.print_color(f"\n🔍 AGGRESSIVE AI SCANNING FOR PAPER TRADES...", self.Fore.BLUE + self.Style.BRIGHT)
                
                for pair in market_data.keys():
                    if pair not in self.paper_positions and len(self.paper_positions) < self.real_bot.max_concurrent_trades:
                        pair_data = {pair: market_data[pair]}
                        decision = self.real_bot.get_ai_decision(pair_data)
                        
                        if decision["action"] == "TRADE":
                            direction_icon = "🟢 LONG" if decision['direction'] == "LONG" else "🔴 SHORT"
                            self.real_bot.print_color(f"🚀 PAPER TRADE SIGNAL: {pair} {direction_icon}", self.Fore.GREEN + self.Style.BRIGHT)
                            self.paper_execute_trade(decision)
            
        except Exception as e:
            self.real_bot.print_color(f"❌ Paper trading error: {e}", self.Fore.RED)

    def start_paper_trading(self):
        self.real_bot.print_color("🎯 STARTING AGGRESSIVE AI PAPER TRADING!", self.Fore.GREEN + self.Style.BRIGHT)
        self.real_bot.print_color("💡 NO REAL MONEY AT RISK - PERFECT FOR TESTING", self.Fore.GREEN)
        self.real_bot.print_color(f"🔥 AGGRESSIVE MODE | Confidence: {self.real_bot.confidence_threshold}%", self.Fore.MAGENTA)
        
        cycle_count = 0
        while True:
            try:
                cycle_count += 1
                self.real_bot.print_color(f"\n🔄 PAPER TRADING CYCLE {cycle_count}", self.Fore.CYAN)
                self.real_bot.print_color("=" * 50, self.Fore.CYAN)
                
                self.run_paper_trading_cycle()
                self.real_bot.print_color(f"⏰ Waiting 20 seconds...", self.Fore.BLUE)
                time.sleep(20)
                
            except KeyboardInterrupt:
                self.real_bot.print_color(f"\n🛑 PAPER TRADING STOPPED", self.Fore.RED + self.Style.BRIGHT)
                
                # Control C နဲ့ရပ်လိုက်ရင်လည်း paper positions တွေဆက်ကြည့်လို့ရအောင်
                while True:
                    try:
                        print("\n" + "="*80)
                        print("🛑 PAPER TRADING STOPPED - OPTIONS")
                        print("="*80)
                        print("1. 📊 View Paper Positions")
                        print("2. 📋 View Paper Trade History") 
                        print("3. 🔄 Restart Paper Trading")
                        print("4. 🚪 Exit to Main Menu")
                        
                        choice = input("Select option (1-4): ").strip()
                        
                        if choice == "1":
                            self.show_paper_positions()
                            self.get_paper_portfolio_status()
                            input("\nPress Enter to continue...")
                        elif choice == "2":
                            # Show paper trade history
                            if self.paper_history:
                                self.real_bot.print_color(f"\n📋 PAPER TRADE HISTORY ({len(self.paper_history)} trades)", self.Fore.CYAN + self.Style.BRIGHT)
                                for i, trade in enumerate(reversed(self.paper_history[-10:])):
                                    pnl = trade.get('pnl', 0)
                                    pnl_color = self.Fore.GREEN if pnl > 0 else self.Fore.RED
                                    direction_icon = "🟢 LONG" if trade['direction'] == 'LONG' else "🔴 SHORT"
                                    self.real_bot.print_color(f"{i+1}. {direction_icon} {trade['pair']} | P&L: ${pnl:.2f}", pnl_color)
                            else:
                                self.real_bot.print_color("No paper trade history", self.Fore.YELLOW)
                            input("\nPress Enter to continue...")
                        elif choice == "3":
                            self.start_paper_trading()  # Restart paper trading
                            break
                        elif choice == "4":
                            break
                        else:
                            print("Invalid choice!")
                            
                    except KeyboardInterrupt:
                        print("\nReturning to main menu...")
                        break
                    except Exception as e:
                        print(f"Error: {e}")
                        time.sleep(2)
                break
                
            except Exception as e:
                self.real_bot.print_color(f"❌ Paper trading error: {e}", self.Fore.RED)
                time.sleep(20)


def main_menu():
    """Main menu for the aggressive bot"""
    bot = AggressiveAIScalpingBot()
    
    while True:
        try:
            print("\n" + "="*80)
            print("🔥 AGGRESSIVE AI TRADING BOT - MAIN MENU")
            print("="*80)
            print("1. 🚀 Start Live Aggressive AI Trading")
            print("2. 📝 Start Paper Aggressive AI Trading") 
            print("3. 📊 View Live Positions Dashboard")
            print("4. 📋 View Trade History")
            print("5. ❌ Exit")
            
            choice = input("Select option (1-5): ").strip()
            
            if choice == "1":
                print("⚠️  WARNING: AGGRESSIVE REAL MONEY TRADING!")
                confirm = input("Type 'AGGRESSIVE' to confirm: ").strip()
                if confirm.upper() == 'AGGRESSIVE':
                    bot.start_trading()
                else:
                    print("Returning to main menu...")
            elif choice == "2":
                paper_bot = AggressiveAIPaperTradingBot(bot)
                paper_bot.start_paper_trading()
            elif choice == "3":
                bot.show_all_positions_dashboard()
                bot.show_bot_managed_positions()
                input("\nPress Enter to continue...")
            elif choice == "4":
                bot.show_trade_history_menu()
            elif choice == "5":
                print("Goodbye! 👋")
                break
            else:
                print("Invalid choice!")
                
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(2)


if __name__ == "__main__":
    try:
        main_menu()
    except Exception as e:
        print(f"❌ Failed to start: {e}")
