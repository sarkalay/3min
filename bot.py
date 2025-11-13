import sys
import os

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Try to import learn script
try:
    from learn_script import SelfLearningAITrader
    LEARN_SCRIPT_AVAILABLE = True
    print("✅ Learn script loaded successfully!")
except ImportError as e:
    print(f"❌ Learn script import failed: {e}")
    LEARN_SCRIPT_AVAILABLE = False

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

# Use conditional inheritance with proper method placement
if LEARN_SCRIPT_AVAILABLE:
    class FullyAutonomous1HourAITrader(SelfLearningAITrader):
        def __init__(self):
            # Initialize learning component first
            super().__init__()
            # Then initialize trading components
            self._initialize_trading()
else:
    class FullyAutonomous1HourAITrader(object):
        def __init__(self):
            # Fallback initialization without learning
            self.mistakes_history = []
            self.learned_patterns = {}
            self.performance_stats = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'common_mistakes': {},
                'improvement_areas': []
            }
            self._initialize_trading()

# Common trading initialization for both cases
def _initialize_trading(self):
    """Initialize trading components (common for both cases)"""
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
    
    # 🎯 FULLY AUTONOMOUS 1HOUR AI TRADING PARAMETERS
    self.total_budget = 500  # $500 budget for AI to manage
    self.available_budget = 500  # Current available budget
    self.max_position_size_percent = 25  # Max 25% of budget per trade for 1hr
    self.max_concurrent_trades = 4  # Maximum concurrent positions
    
    # AI can trade selected 4 major pairs only
    self.available_pairs = [
        "BTCUSDT", "BNBUSDT", "SOLUSDT", "AVAXUSDT"
    ]
    
    # Track AI-opened trades
    self.ai_opened_trades = {}
    
    # REAL TRADE HISTORY
    self.real_trade_history_file = "fully_autonomous_1hour_ai_trading_history.json"
    self.real_trade_history = self.load_real_trade_history()
    
    # Trading statistics
    self.real_total_trades = 0
    self.real_winning_trades = 0
    self.real_total_pnl = 0.0
    
    # Precision settings
    self.quantity_precision = {}
    self.price_precision = {}
    
    # NEW: Reverse position settings
    self.allow_reverse_positions = True  # Enable reverse position feature
    
    # NEW: Monitoring interval (3 minute)
    self.monitoring_interval = 180  # 3 minute in seconds
    
    # Initialize Binance client
    try:
        self.binance = Client(self.binance_api_key, self.binance_secret)
        self.print_color(f"🤖 FULLY AUTONOMOUS AI TRADER ACTIVATED! 🤖", self.Fore.CYAN + self.Style.BRIGHT)
        self.print_color(f"💰 TOTAL BUDGET: ${self.total_budget}", self.Fore.GREEN + self.Style.BRIGHT)
        self.print_color(f"🔄 REVERSE POSITION FEATURE: ENABLED", self.Fore.MAGENTA + self.Style.BRIGHT)
        self.print_color(f"🎯 NO TP/SL - AI MANUAL CLOSE ONLY", self.Fore.YELLOW + self.Style.BRIGHT)
        self.print_color(f"⏰ MONITORING: 3 MINUTE INTERVAL", self.Fore.RED + self.Style.BRIGHT)
        self.print_color(f"📊 Max Positions: {self.max_concurrent_trades}", self.Fore.YELLOW + self.Style.BRIGHT)
        if LEARN_SCRIPT_AVAILABLE:
            self.print_color(f"🧠 SELF-LEARNING AI: ENABLED", self.Fore.MAGENTA + self.Style.BRIGHT)
    except Exception as e:
        self.print_color(f"Binance initialization failed: {e}", self.Fore.RED)
        self.binance = None
    
    self.validate_config()
    if self.binance:
        self.setup_futures()
        self.load_symbol_precision()

# Add the method to both classes
FullyAutonomous1HourAITrader._initialize_trading = _initialize_trading

# Now add all the other methods to the class
def load_real_trade_history(self):
    """Load trading history"""
    try:
        if os.path.exists(self.real_trade_history_file):
            with open(self.real_trade_history_file, 'r') as f:
                history = json.load(f)
                self.real_total_trades = len(history)
                self.real_winning_trades = len([t for t in history if t.get('pnl', 0) > 0])
                self.real_total_pnl = sum(t.get('pnl', 0) for t in history)
                return history
        return []
    except Exception as e:
        self.print_color(f"Error loading trade history: {e}", self.Fore.RED)
        return []

def save_real_trade_history(self):
    """Save trading history"""
    try:
        with open(self.real_trade_history_file, 'w') as f:
            json.dump(self.real_trade_history, f, indent=2)
    except Exception as e:
        self.print_color(f"Error saving trade history: {e}", self.Fore.RED)

def add_trade_to_history(self, trade_data):
    """Add trade to history WITH learning"""
    try:
        trade_data['close_time'] = self.get_thailand_time()
        trade_data['close_timestamp'] = time.time()
        trade_data['trade_type'] = 'REAL'
        self.real_trade_history.append(trade_data)
        
        # 🧠 Learn from this trade (especially if it's a loss)
        if LEARN_SCRIPT_AVAILABLE:
            self.learn_from_mistake(trade_data)
        
        # Update performance stats
        self.performance_stats['total_trades'] += 1
        pnl = trade_data.get('pnl', 0)
        self.real_total_pnl += pnl
        if pnl > 0:
            self.real_winning_trades += 1
            self.performance_stats['winning_trades'] += 1
        else:
            self.performance_stats['losing_trades'] += 1
            
        if len(self.real_trade_history) > 200:
            self.real_trade_history = self.real_trade_history[-200:]
        self.save_real_trade_history()
        self.print_color(f"📝 Trade saved: {trade_data['pair']} {trade_data['direction']} P&L: ${pnl:.2f}", self.Fore.CYAN)
    except Exception as e:
        self.print_color(f"Error adding trade to history: {e}", self.Fore.RED)

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
            self.print_color("✅ Binance connection successful!", self.Fore.GREEN + self.Style.BRIGHT)
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
                # Set initial leverage to 10x (AI can change later)
                self.binance.futures_change_leverage(symbol=pair, leverage=10)
                self.binance.futures_change_margin_type(symbol=pair, marginType='ISOLATED')
                self.print_color(f"✅ Leverage set for {pair}", self.Fore.GREEN)
            except Exception as e:
                self.print_color(f"Leverage setup failed for {pair}: {e}", self.Fore.YELLOW)
        self.print_color("✅ Futures setup completed!", self.Fore.GREEN + self.Style.BRIGHT)
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
        self.print_color("✅ Symbol precision loaded", self.Fore.GREEN + self.Style.BRIGHT)
    except Exception as e:
        self.print_color(f"Error loading symbol precision: {e}", self.Fore.RED)

def get_market_news_sentiment(self):
    """Get recent cryptocurrency news sentiment"""
    try:
        news_sources = [
            "CoinDesk", "Cointelegraph", "CryptoSlate", "Decrypt", "Binance Blog"
        ]
        return f"Monitoring: {', '.join(news_sources)}"
    except:
        return "General crypto market news monitoring"

def get_ai_trading_decision(self, pair, market_data, current_trade=None):
    """AI makes COMPLETE trading decisions including REVERSE positions"""
    try:
        if not self.openrouter_key:
            return self.get_fallback_decision(pair, market_data)
        
        current_price = market_data['current_price']
        
        # NEW: Check if we have existing trade for reverse position analysis
        reverse_analysis = ""
        if current_trade and self.allow_reverse_positions:
            current_pnl = self.calculate_current_pnl(current_trade, current_price)
            reverse_analysis = f"""
            EXISTING POSITION ANALYSIS:
            - Current Position: {current_trade['direction']}
            - Entry Price: ${current_trade['entry_price']}
            - Current PnL: {current_pnl:.2f}%
            - Should we REVERSE this position?
            """
        
        # 🧠 Add learning context to prompt
        learning_context = ""
        if LEARN_SCRIPT_AVAILABLE and hasattr(self, 'get_learning_enhanced_prompt'):
            learning_context = self.get_learning_enhanced_prompt(pair, market_data)
        
        # 🧠 COMPREHENSIVE AI TRADING PROMPT WITH REVERSE FEATURE
        prompt = f"""
        YOU ARE A FULLY AUTONOMOUS AI TRADER with ${self.available_budget:.2f} budget.

        {learning_context}

        MARKET ANALYSIS FOR {pair} (3MINUTE MONITORING):
        - Current Price: ${current_price:.6f}
        - 1Hour Price Change: {market_data.get('price_change', 0):.2f}%
        - Support/Resistance Levels: {market_data.get('support_levels', [])} / {market_data.get('resistance_levels', [])}
        {reverse_analysis}

        IMPORTANT: NO TP/SL ORDERS WILL BE SET!
        - You must manually monitor and close positions
        - Consider market conditions for entry AND exit
        - Close positions based on trend changes, not fixed levels

        REVERSE POSITION STRATEGY:
        - If existing position is losing and market trend reversed, consider REVERSE
        - Close current position and open opposite direction immediately

        Return VALID JSON only:
        {{
            "decision": "LONG" | "SHORT" | "HOLD" | "REVERSE_LONG" | "REVERSE_SHORT",
            "position_size_usd": number,
            "entry_price": number,
            "leverage": number (10-30),
            "confidence": 0-100,
            "reasoning": "analysis including when to manually close based on market conditions"
        }}

        REVERSE Decisions meaning:
        - "REVERSE_LONG": Close SHORT position (if any), open LONG immediately  
        - "REVERSE_SHORT": Close LONG position (if any), open SHORT immediately
        """

        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com",
            "X-Title": "Fully Autonomous AI Trader"
        }
        
        data = {
            "model": "deepseek/deepseek-chat-v3.1",
            "messages": [
                {"role": "system", "content": "You are a fully autonomous AI trader with reverse position capability. You manually close positions based on market conditions - no TP/SL orders are set. Analyze when to enter AND when to exit based on technical analysis. Monitor every 1 minute."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 800
        }
        
        self.print_color(f"🧠 DeepSeek Analyzing {pair} with 3MIN monitoring...", self.Fore.MAGENTA + self.Style.BRIGHT)
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result['choices'][0]['message']['content'].strip()
            return self.parse_ai_trading_decision(ai_response, pair, current_price, current_trade)
        else:
            self.print_color(f"DeepSeek API error: {response.status_code}", self.Fore.RED)
            return self.get_fallback_decision(pair, market_data)
            
    except Exception as e:
        self.print_color(f"DeepSeek analysis failed: {e}", self.Fore.RED)
        return self.get_fallback_decision(pair, market_data)

def parse_ai_trading_decision(self, ai_response, pair, current_price, current_trade=None):
    """Parse AI's trading decision including REVERSE positions"""
    try:
        json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            decision_data = json.loads(json_str)
            
            decision = decision_data.get('decision', 'HOLD').upper()
            position_size_usd = float(decision_data.get('position_size_usd', 0))
            entry_price = float(decision_data.get('entry_price', 0))
            leverage = int(decision_data.get('leverage', 10))
            confidence = float(decision_data.get('confidence', 50))
            reasoning = decision_data.get('reasoning', 'AI Analysis')
            
            # Validate leverage
            if leverage < 10:
                leverage = 10
            elif leverage > 30:
                leverage = 30
                
            if entry_price <= 0:
                entry_price = current_price
                
            return {
                "decision": decision,
                "position_size_usd": position_size_usd,
                "entry_price": entry_price,
                "leverage": leverage,
                "confidence": confidence,
                "reasoning": reasoning,
                "should_reverse": decision.startswith('REVERSE_')
            }
        return self.get_fallback_decision(pair, {'current_price': current_price})
    except Exception as e:
        self.print_color(f"DeepSeek response parsing failed: {e}", self.Fore.RED)
        return self.get_fallback_decision(pair, {'current_price': current_price})

def get_fallback_decision(self, pair, market_data):
    """Fallback decision if AI fails"""
    return {
        "decision": "HOLD",
        "position_size_usd": 0,
        "entry_price": market_data['current_price'],
        "leverage": 10,
        "confidence": 0,
        "reasoning": "Fallback: AI analysis unavailable",
        "should_reverse": False
    }

def calculate_current_pnl(self, trade, current_price):
    """Calculate current PnL percentage"""
    try:
        if trade['direction'] == 'LONG':
            pnl_percent = ((current_price - trade['entry_price']) / trade['entry_price']) * 100 * trade['leverage']
        else:
            pnl_percent = ((trade['entry_price'] - current_price) / trade['entry_price']) * 100 * trade['leverage']
        return pnl_percent
    except:
        return 0

def execute_reverse_position(self, pair, ai_decision, current_trade):
    """Execute reverse position - CLOSE CURRENT, THEN ASK AI BEFORE OPENING REVERSE"""
    try:
        self.print_color(f"🔄 ATTEMPTING REVERSE POSITION FOR {pair}", self.Fore.YELLOW + self.Style.BRIGHT)
        
        # 1. First close the current losing position
        close_success = self.close_trade_immediately(pair, current_trade, "REVERSE_POSITION")
        
        if close_success:
            # 2. Wait a moment for position to close
            time.sleep(2)
            
            # 3. Verify position is actually removed
            if pair in self.ai_opened_trades:
                self.print_color(f"⚠️  Position still exists after close, forcing removal...", self.Fore.RED)
                del self.ai_opened_trades[pair]
            
            # 4. 🆕 ASK AI AGAIN BEFORE OPENING REVERSE POSITION
            self.print_color(f"🔍 Asking AI to confirm reverse position for {pair}...", self.Fore.BLUE)
            market_data = self.get_price_history(pair)
            
            # Get fresh AI decision after closing
            new_ai_decision = self.get_ai_trading_decision(pair, market_data, None)
            
            # Check if AI still wants to open reverse position
            if new_ai_decision["decision"] in ["LONG", "SHORT"] and new_ai_decision["position_size_usd"] > 0:
                # 🎯 Calculate correct reverse direction
                current_direction = current_trade['direction']
                if current_direction == "LONG":
                    correct_reverse_direction = "SHORT"
                else:
                    correct_reverse_direction = "LONG"
                
                self.print_color(f"✅ AI CONFIRMED: Opening {correct_reverse_direction} {pair}", self.Fore.CYAN + self.Style.BRIGHT)
                
                # Use the new AI decision but ensure correct direction
                reverse_decision = new_ai_decision.copy()
                reverse_decision["decision"] = correct_reverse_direction
                
                # Execute the reverse trade
                return self.execute_ai_trade(pair, reverse_decision)
            else:
                self.print_color(f"🔄 AI changed mind, not opening reverse position for {pair}", self.Fore.YELLOW)
                self.print_color(f"📝 AI Decision: {new_ai_decision['decision']} | Reason: {new_ai_decision['reasoning']}", self.Fore.WHITE)
                return False
        else:
            self.print_color(f"❌ Reverse position failed: Could not close current trade", self.Fore.RED)
            return False
            
    except Exception as e:
        self.print_color(f"❌ Reverse position execution failed: {e}", self.Fore.RED)
        return False

def close_trade_immediately(self, pair, trade, reason="REVERSE"):
    """Close trade immediately at market price"""
    try:
        if self.binance:
            # Cancel any existing orders first
            try:
                open_orders = self.binance.futures_get_open_orders(symbol=pair)
                for order in open_orders:
                    if order['reduceOnly']:
                        self.binance.futures_cancel_order(symbol=pair, orderId=order['orderId'])
            except Exception as e:
                self.print_color(f"Order cancel warning: {e}", self.Fore.YELLOW)
            
            # Close position with market order
            close_side = 'SELL' if trade['direction'] == 'LONG' else 'BUY'
            order = self.binance.futures_create_order(
                symbol=pair,
                side=close_side,
                type='MARKET',
                quantity=abs(trade['quantity']),
                reduceOnly=True
            )
            
            # Calculate final PnL
            current_price = self.get_current_price(pair)
            if trade['direction'] == 'LONG':
                pnl = (current_price - trade['entry_price']) * trade['quantity']
            else:
                pnl = (trade['entry_price'] - current_price) * trade['quantity']
            
            # Update trade record
            trade['status'] = 'CLOSED'
            trade['exit_price'] = current_price
            trade['pnl'] = pnl
            trade['close_reason'] = reason
            trade['close_time'] = self.get_thailand_time()
            
            # Return budget
            self.available_budget += trade['position_size_usd'] + pnl
            
            self.add_trade_to_history(trade.copy())
            self.print_color(f"✅ Position closed for reverse: {pair} | P&L: ${pnl:.2f}", self.Fore.CYAN)
            
            # Remove from active positions after closing
            if pair in self.ai_opened_trades:
                del self.ai_opened_trades[pair]
            
            return True
        else:
            # Paper trading close
            current_price = self.get_current_price(pair)
            if trade['direction'] == 'LONG':
                pnl = (current_price - trade['entry_price']) * trade['quantity']
            else:
                pnl = (trade['entry_price'] - current_price) * trade['quantity']
            
            trade['status'] = 'CLOSED'
            trade['exit_price'] = current_price
            trade['pnl'] = pnl
            trade['close_reason'] = reason
            trade['close_time'] = self.get_thailand_time()
            
            self.available_budget += trade['position_size_usd'] + pnl
            self.add_trade_to_history(trade.copy())
            
            # Remove from active positions after closing
            if pair in self.ai_opened_trades:
                del self.ai_opened_trades[pair]
            
            return True
            
    except Exception as e:
        self.print_color(f"❌ Immediate close failed: {e}", self.Fore.RED)
        return False

def get_price_history(self, pair, limit=12):
    """Get 1hour price history with technical levels"""
    try:
        if self.binance:
            klines = self.binance.futures_klines(symbol=pair, interval=Client.KLINE_INTERVAL_1HOUR, limit=limit)
            prices = [float(k[4]) for k in klines]
            highs = [float(k[2]) for k in klines]
            lows = [float(k[3]) for k in klines]
            volumes = [float(k[5]) for k in klines]
            
            current_price = prices[-1] if prices else 0
            # Calculate 4-hour price change for 1hr context
            price_change = ((current_price - prices[-4]) / prices[-4] * 100) if len(prices) >= 4 else 0
            volume_change = ((volumes[-1] - volumes[-4]) / volumes[-4] * 100) if len(volumes) >= 4 else 0
            
            # Calculate support/resistance levels for 1hr
            support_levels = [min(lows[-6:]), min(lows[-12:])]
            resistance_levels = [max(highs[-6:]), max(highs[-12:])]
            
            return {
                'prices': prices,
                'highs': highs,
                'lows': lows,
                'volumes': volumes,
                'current_price': current_price,
                'price_change': price_change,
                'volume_change': volume_change,
                'support_levels': [round(l, 4) for l in support_levels],
                'resistance_levels': [round(l, 4) for l in resistance_levels]
            }
        else:
            current_price = self.get_current_price(pair)
            return {
                'prices': [current_price] * 12,
                'highs': [current_price * 1.03] * 12,
                'lows': [current_price * 0.97] * 12,
                'volumes': [100000] * 12,
                'current_price': current_price,
                'price_change': 1.2,
                'volume_change': 15.5,
                'support_levels': [current_price * 0.97, current_price * 0.95],
                'resistance_levels': [current_price * 1.03, current_price * 1.05]
            }
    except Exception as e:
        current_price = self.get_current_price(pair)
        return {
            'current_price': current_price,
            'price_change': 0,
            'volume_change': 0,
            'support_levels': [],
            'resistance_levels': []
        }

def get_current_price(self, pair):
    try:
        if self.binance:
            ticker = self.binance.futures_symbol_ticker(symbol=pair)
            return float(ticker['price'])
        else:
            # Mock prices for paper trading
            mock_prices = {
                "BTCUSDT": 45000, "BNBUSDT": 300,
                "SOLUSDT": 180, "AVAXUSDT": 35
            }
            return mock_prices.get(pair, 100)
    except:
        return 100

def calculate_quantity(self, pair, entry_price, position_size_usd, leverage):
    """Calculate quantity based on position size and leverage"""
    try:
        if entry_price <= 0:
            return None
            
        # Calculate notional value
        notional_value = position_size_usd * leverage
        
        # Calculate quantity
        quantity = notional_value / entry_price
        
        # Apply precision
        precision = self.quantity_precision.get(pair, 3)
        quantity = round(quantity, precision)
        
        if quantity <= 0:
            return None
            
        self.print_color(f"📊 Position: ${position_size_usd} | Leverage: {leverage}x | Notional: ${notional_value:.2f} | Quantity: {quantity}", self.Fore.CYAN)
        return quantity
        
    except Exception as e:
        self.print_color(f"Quantity calculation failed: {e}", self.Fore.RED)
        return None

def can_open_new_position(self, pair, position_size_usd):
    """Check if new position can be opened"""
    if pair in self.ai_opened_trades:
        return False, "Position already exists"
    
    if len(self.ai_opened_trades) >= self.max_concurrent_trades:
        return False, f"Max concurrent trades reached ({self.max_concurrent_trades})"
        
    if position_size_usd > self.available_budget:
        return False, f"Insufficient budget: ${position_size_usd:.2f} > ${self.available_budget:.2f}"
        
    max_allowed = self.total_budget * self.max_position_size_percent / 100
    if position_size_usd > max_allowed:
        return False, f"Position size too large: ${position_size_usd:.2f} > ${max_allowed:.2f}"
        
    return True, "OK"

def get_ai_decision_with_learning(self, pair, market_data):
    """Get AI decision enhanced with learned lessons"""
    # First get normal AI decision
    ai_decision = self.get_ai_trading_decision(pair, market_data)
    
    # Check if this matches known mistake patterns
    if LEARN_SCRIPT_AVAILABLE and hasattr(self, 'should_avoid_trade') and self.should_avoid_trade(ai_decision, market_data):
        self.print_color(f"🧠 AI USING LEARNING: Blocking potential mistake for {pair}", self.Fore.YELLOW)
        return {
            "decision": "HOLD",
            "position_size_usd": 0,
            "entry_price": market_data['current_price'],
            "leverage": 10,
            "confidence": 0,
            "reasoning": f"Blocked - matches known error pattern",
            "should_reverse": False
        }
    
    # Add learning context to reasoning
    if ai_decision["decision"] != "HOLD" and LEARN_SCRIPT_AVAILABLE and hasattr(self, 'mistakes_history'):
        learning_context = f" | Applying lessons from {len(self.mistakes_history)} past mistakes"
        ai_decision["reasoning"] += learning_context
    
    return ai_decision

def execute_ai_trade(self, pair, ai_decision):
    """Execute trade WITHOUT TP/SL orders - AI will close manually"""
    try:
        decision = ai_decision["decision"]
        position_size_usd = ai_decision["position_size_usd"]
        entry_price = ai_decision["entry_price"]
        leverage = ai_decision["leverage"]
        confidence = ai_decision["confidence"]
        reasoning = ai_decision["reasoning"]
        
        # NEW: Handle reverse positions
        if decision.startswith('REVERSE_'):
            if pair in self.ai_opened_trades:
                current_trade = self.ai_opened_trades[pair]
                return self.execute_reverse_position(pair, ai_decision, current_trade)
            else:
                self.print_color(f"❌ Cannot reverse: No active position for {pair}", self.Fore.RED)
                return False
        
        if decision == "HOLD" or position_size_usd <= 0:
            self.print_color(f"🟡 DeepSeek decides to HOLD {pair}", self.Fore.YELLOW)
            return False
        
        # Check if we can open position (skip if reversing)
        if pair in self.ai_opened_trades and not decision.startswith('REVERSE_'):
            self.print_color(f"🚫 Cannot open {pair}: Position already exists", self.Fore.RED)
            return False
        
        if len(self.ai_opened_trades) >= self.max_concurrent_trades and pair not in self.ai_opened_trades:
            self.print_color(f"🚫 Cannot open {pair}: Max concurrent trades reached", self.Fore.RED)
            return False
            
        if position_size_usd > self.available_budget:
            self.print_color(f"🚫 Cannot open {pair}: Insufficient budget", self.Fore.RED)
            return False
        
        # Calculate quantity
        quantity = self.calculate_quantity(pair, entry_price, position_size_usd, leverage)
        if quantity is None:
            return False
        
        # Display AI trade decision (NO TP/SL)
        direction_color = self.Fore.GREEN + self.Style.BRIGHT if decision == 'LONG' else self.Fore.RED + self.Style.BRIGHT
        direction_icon = "🟢 LONG" if decision == 'LONG' else "🔴 SHORT"
        
        self.print_color(f"\n🤖 DEEPSEEK TRADE EXECUTION (NO TP/SL)", self.Fore.CYAN + self.Style.BRIGHT)
        self.print_color("=" * 80, self.Fore.CYAN)
        self.print_color(f"{direction_icon} {pair}", direction_color)
        self.print_color(f"POSITION SIZE: ${position_size_usd:.2f}", self.Fore.GREEN + self.Style.BRIGHT)
        self.print_color(f"LEVERAGE: {leverage}x ⚡", self.Fore.RED + self.Style.BRIGHT)
        self.print_color(f"ENTRY PRICE: ${entry_price:.4f}", self.Fore.WHITE)
        self.print_color(f"QUANTITY: {quantity}", self.Fore.CYAN)
        self.print_color(f"🎯 NO TP/SL SET - AI WILL CLOSE MANUALLY BASED ON MARKET", self.Fore.YELLOW + self.Style.BRIGHT)
        self.print_color(f"CONFIDENCE: {confidence}%", self.Fore.YELLOW + self.Style.BRIGHT)
        self.print_color(f"REASONING: {reasoning}", self.Fore.WHITE)
        self.print_color("=" * 80, self.Fore.CYAN)
        
        # Execute live trade WITHOUT TP/SL orders
        if self.binance:
            entry_side = 'BUY' if decision == 'LONG' else 'SELL'
            
            # Set leverage
            try:
                self.binance.futures_change_leverage(symbol=pair, leverage=leverage)
            except Exception as e:
                self.print_color(f"Leverage change failed: {e}", self.Fore.YELLOW)
            
            # Execute order ONLY - no TP/SL orders
            order = self.binance.futures_create_order(
                symbol=pair,
                side=entry_side,
                type='MARKET',
                quantity=quantity
            )
            
            # ❌❌❌ NO TP/SL ORDERS CREATED ❌❌❌
        
        # Update budget and track trade
        self.available_budget -= position_size_usd
        
        self.ai_opened_trades[pair] = {
            "pair": pair,
            "direction": decision,
            "entry_price": entry_price,
            "quantity": quantity,
            "position_size_usd": position_size_usd,
            "leverage": leverage,
            "entry_time": time.time(),
            "status": 'ACTIVE',
            'ai_confidence': confidence,
            'ai_reasoning': reasoning,
            'entry_time_th': self.get_thailand_time(),
            'has_tp_sl': False  # NEW: Mark as no TP/SL
        }
        
        self.print_color(f"✅ TRADE EXECUTED (NO TP/SL): {pair} {decision} | Leverage: {leverage}x", self.Fore.GREEN + self.Style.BRIGHT)
        self.print_color(f"📊 AI will monitor and close manually based on market conditions", self.Fore.BLUE)
        return True
        
    except Exception as e:
        self.print_color(f"❌ Trade execution failed: {e}", self.Fore.RED)
        return False

# Add all methods to the class
methods = [
    load_real_trade_history, save_real_trade_history, add_trade_to_history,
    get_thailand_time, print_color, validate_config, setup_futures,
    load_symbol_precision, get_market_news_sentiment, get_ai_trading_decision,
    parse_ai_trading_decision, get_fallback_decision, calculate_current_pnl,
    execute_reverse_position, close_trade_immediately, get_price_history,
    get_current_price, calculate_quantity, can_open_new_position,
    get_ai_decision_with_learning, execute_ai_trade
]

for method in methods:
    setattr(FullyAutonomous1HourAITrader, method.__name__, method)

# Add the remaining methods
def get_ai_close_decision(self, pair, trade):
    """Ask AI whether to close this position based on market conditions"""
    try:
        current_price = self.get_current_price(pair)
        market_data = self.get_price_history(pair)
        current_pnl = self.calculate_current_pnl(trade, current_price)
        
        prompt = f"""
        SHOULD WE CLOSE THIS POSITION? (3MINUTE MONITORING)
        
        CURRENT ACTIVE TRADE:
        - Pair: {pair}
        - Direction: {trade['direction']}
        - Entry Price: ${trade['entry_price']:.4f}
        - Current Price: ${current_price:.4f}
        - PnL: {current_pnl:.2f}%
        - Position Size: ${trade['position_size_usd']:.2f}
        - Leverage: {trade['leverage']}x
        - Trade Age: {(time.time() - trade['entry_time']) / 60:.1f} minutes
        
        MARKET CONDITIONS:
        - 1H Change: {market_data.get('price_change', 0):.2f}%
        - Support: {market_data.get('support_levels', [])}
        - Resistance: {market_data.get('resistance_levels', [])}
        - Current Trend: {'BULLISH' if market_data.get('price_change', 0) > 0 else 'BEARISH'}
        
        Should we CLOSE this position now?
        Consider:
        - Profit/loss situation
        - Trend changes and momentum
        - Technical indicators
        - Market sentiment
        - Risk management
        - Time in trade
        
        Return JSON:
        {{
            "should_close": true/false,
            "close_reason": "TAKE_PROFIT" | "STOP_LOSS" | "TREND_REVERSAL" | "TIME_EXIT" | "MARKET_CONDITION",
            "confidence": 0-100,
            "reasoning": "Detailed technical analysis for close decision"
        }}
        """
        
        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com",
            "X-Title": "Fully Autonomous AI Trader"
        }
        
        data = {
            "model": "deepseek/deepseek-chat-v3.1",
            "messages": [
                {"role": "system", "content": "You are an AI trader monitoring active positions every 1 minute. Decide whether to close positions based on current market conditions, technical analysis, and risk management. Provide clear reasoning for your close decisions."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 600
        }
        
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=45)
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result['choices'][0]['message']['content'].strip()
            
            # Parse AI response
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                close_decision = json.loads(json_str)
                return close_decision
                
        return {"should_close": False, "close_reason": "AI_UNAVAILABLE", "confidence": 0, "reasoning": "AI analysis failed"}
        
    except Exception as e:
        self.print_color(f"AI close decision error: {e}", self.Fore.RED)
        return {"should_close": False, "close_reason": "ERROR", "confidence": 0, "reasoning": f"Error: {e}"}

def monitor_positions(self):
    """Monitor positions and ask AI when to close (NO TP/SL)"""
    try:
        closed_trades = []
        for pair, trade in list(self.ai_opened_trades.items()):
            if trade['status'] != 'ACTIVE':
                continue
            
            # NEW: Ask AI whether to close this position (for positions without TP/SL)
            if not trade.get('has_tp_sl', True):
                self.print_color(f"🔍 Checking if AI wants to close {pair}...", self.Fore.BLUE)
                close_decision = self.get_ai_close_decision(pair, trade)
                
                if close_decision.get("should_close", False):
                    close_reason = close_decision.get("close_reason", "AI_DECISION")
                    confidence = close_decision.get("confidence", 0)
                    reasoning = close_decision.get("reasoning", "No reason provided")
                    
                    self.print_color(f"🎯 AI Decision: CLOSE {pair} (Confidence: {confidence}%)", self.Fore.YELLOW + self.Style.BRIGHT)
                    self.print_color(f"📝 Reason: {reasoning}", self.Fore.WHITE)
                    
                    success = self.close_trade_immediately(pair, trade, f"AI_CLOSE: {close_reason}")
                    if success:
                        closed_trades.append(pair)
                else:
                    # Show AI's decision to hold
                    if close_decision.get('confidence', 0) > 0:
                        self.print_color(f"🔍 AI wants to HOLD {pair} (Confidence: {close_decision.get('confidence', 0)}%)", self.Fore.GREEN)
                
        return closed_trades
                
    except Exception as e:
        self.print_color(f"Monitoring error: {e}", self.Fore.RED)
        return []

def display_dashboard(self):
    """Display trading dashboard WITH learning progress"""
    self.print_color(f"\n🤖 AI TRADING DASHBOARD - {self.get_thailand_time()}", self.Fore.CYAN + self.Style.BRIGHT)
    self.print_color("=" * 90, self.Fore.CYAN)
    self.print_color(f"🎯 MODE: NO TP/SL - AI MANUAL CLOSE ONLY", self.Fore.YELLOW + self.Style.BRIGHT)
    self.print_color(f"⏰ MONITORING: 1 MINUTE INTERVAL", self.Fore.RED + self.Style.BRIGHT)
    
    # 🧠 Add learning stats
    if LEARN_SCRIPT_AVAILABLE and hasattr(self, 'mistakes_history'):
        total_lessons = len(self.mistakes_history)
        if total_lessons > 0:
            self.print_color(f"🧠 AI HAS LEARNED FROM {total_lessons} MISTAKES", self.Fore.MAGENTA + self.Style.BRIGHT)
    
    active_count = 0
    total_unrealized = 0
    
    for pair, trade in self.ai_opened_trades.items():
        if trade['status'] == 'ACTIVE':
            active_count += 1
            current_price = self.get_current_price(pair)
            
            direction_icon = "🟢 LONG" if trade['direction'] == 'LONG' else "🔴 SHORT"
            
            if trade['direction'] == 'LONG':
                unrealized_pnl = (current_price - trade['entry_price']) * trade['quantity']
            else:
                unrealized_pnl = (trade['entry_price'] - current_price) * trade['quantity']
                
            total_unrealized += unrealized_pnl
            pnl_color = self.Fore.GREEN + self.Style.BRIGHT if unrealized_pnl >= 0 else self.Fore.RED + self.Style.BRIGHT
            
            self.print_color(f"{direction_icon} {pair}", self.Fore.WHITE + self.Style.BRIGHT)
            self.print_color(f"   Size: ${trade['position_size_usd']:.2f} | Leverage: {trade['leverage']}x ⚡", self.Fore.WHITE)
            self.print_color(f"   Entry: ${trade['entry_price']:.4f} | Current: ${current_price:.4f}", self.Fore.WHITE)
            self.print_color(f"   P&L: ${unrealized_pnl:.2f}", pnl_color)
            self.print_color(f"   🎯 NO TP/SL - AI Monitoring Every 3min", self.Fore.YELLOW)
            self.print_color("   " + "-" * 60, self.Fore.CYAN)
    
    if active_count == 0:
        self.print_color("No active positions", self.Fore.YELLOW)
    else:
        total_color = self.Fore.GREEN + self.Style.BRIGHT if total_unrealized >= 0 else self.Fore.RED + self.Style.BRIGHT
        self.print_color(f"📊 Active Positions: {active_count}/{self.max_concurrent_trades} | Total Unrealized P&L: ${total_unrealized:.2f}", total_color)

def show_trade_history(self, limit=15):
    """Show trading history"""
    if not self.real_trade_history:
        self.print_color("No trade history found", self.Fore.YELLOW)
        return
    
    self.print_color(f"\n📊 TRADING HISTORY (Last {min(limit, len(self.real_trade_history))} trades)", self.Fore.CYAN + self.Style.BRIGHT)
    self.print_color("=" * 120, self.Fore.CYAN)
    
    recent_trades = self.real_trade_history[-limit:]
    for i, trade in enumerate(reversed(recent_trades)):
        pnl = trade.get('pnl', 0)
        pnl_color = self.Fore.GREEN + self.Style.BRIGHT if pnl > 0 else self.Fore.RED + self.Style.BRIGHT if pnl < 0 else self.Fore.YELLOW
        direction_icon = "🟢 LONG" if trade['direction'] == 'LONG' else "🔴 SHORT"
        position_size = trade.get('position_size_usd', 0)
        leverage = trade.get('leverage', 1)
        
        self.print_color(f"{i+1:2d}. {direction_icon} {trade['pair']} | Size: ${position_size:.2f} | Leverage: {leverage}x | P&L: ${pnl:.2f}", pnl_color)
        self.print_color(f"     Entry: ${trade.get('entry_price', 0):.4f} | Exit: ${trade.get('exit_price', 0):.4f} | {trade.get('close_reason', 'N/A')}", self.Fore.YELLOW)

def show_trading_stats(self):
    """Show trading statistics"""
    if self.real_total_trades == 0:
        return
        
    win_rate = (self.real_winning_trades / self.real_total_trades) * 100
    avg_trade = self.real_total_pnl / self.real_total_trades
    
    self.print_color(f"\n📈 TRADING STATISTICS", self.Fore.GREEN + self.Style.BRIGHT)
    self.print_color("=" * 60, self.Fore.GREEN)
    self.print_color(f"Total Trades: {self.real_total_trades} | Winning Trades: {self.real_winning_trades}", self.Fore.WHITE)
    self.print_color(f"Win Rate: {win_rate:.1f}%", self.Fore.GREEN + self.Style.BRIGHT if win_rate > 50 else self.Fore.YELLOW)
    self.print_color(f"Total P&L: ${self.real_total_pnl:.2f}", self.Fore.GREEN + self.Style.BRIGHT if self.real_total_pnl > 0 else self.Fore.RED + self.Style.BRIGHT)
    self.print_color(f"Average P&L per Trade: ${avg_trade:.2f}", self.Fore.WHITE)
    self.print_color(f"Available Budget: ${self.available_budget:.2f}", self.Fore.CYAN + self.Style.BRIGHT)

def run_trading_cycle(self):
    """Run trading cycle with REVERSE position checking and AI manual close"""
    try:
        # First monitor and ask AI to close positions
        self.monitor_positions()
        self.display_dashboard()
        
        # Show stats periodically
        if hasattr(self, 'cycle_count') and self.cycle_count % 5 == 0:  # Every 5 cycles (5 minutes)
            self.show_trade_history(8)
            self.show_trading_stats()
        
        self.print_color(f"\n🔍 DEEPSEEK SCANNING {len(self.available_pairs)} PAIRS...", self.Fore.BLUE + self.Style.BRIGHT)
        
        qualified_signals = 0
        for pair in self.available_pairs:
            if self.available_budget > 100:
                market_data = self.get_price_history(pair)
                
                # Use learning-enhanced AI decision
                ai_decision = self.get_ai_decision_with_learning(pair, market_data)
                
                if ai_decision["decision"] != "HOLD" and ai_decision["position_size_usd"] > 0:
                    qualified_signals += 1
                    direction = ai_decision['decision']
                    leverage_info = f"Leverage: {ai_decision['leverage']}x"
                    
                    if direction.startswith('REVERSE_'):
                        self.print_color(f"🔄 REVERSE SIGNAL: {pair} {direction} | Size: ${ai_decision['position_size_usd']:.2f}", self.Fore.YELLOW + self.Style.BRIGHT)
                    else:
                        self.print_color(f"🎯 TRADE SIGNAL: {pair} {direction} | Size: ${ai_decision['position_size_usd']:.2f} | {leverage_info}", self.Fore.GREEN + self.Style.BRIGHT)
                    
                    success = self.execute_ai_trade(pair, ai_decision)
                    if success:
                        time.sleep(2)  # Reduced delay for faster 3min cycles
            
        if qualified_signals == 0:
            self.print_color("No qualified DeepSeek signals this cycle", self.Fore.YELLOW)
            
    except Exception as e:
        self.print_color(f"Trading cycle error: {e}", self.Fore.RED)

def start_trading(self):
    """Start trading with REVERSE position feature and NO TP/SL"""
    self.print_color("🚀 STARTING AI TRADER WITH 3MINUTE MONITORING!", self.Fore.CYAN + self.Style.BRIGHT)
    self.print_color("💰 AI MANAGING $500 PORTFOLIO", self.Fore.GREEN + self.Style.BRIGHT)
    self.print_color("🔄 REVERSE POSITION: ENABLED (AI can flip losing positions)", self.Fore.MAGENTA + self.Style.BRIGHT)
    self.print_color("🎯 NO TP/SL - AI MANUAL CLOSE ONLY", self.Fore.YELLOW + self.Style.BRIGHT)
    self.print_color("⏰ MONITORING: 1 MINUTE INTERVAL", self.Fore.RED + self.Style.BRIGHT)
    self.print_color("⚡ LEVERAGE: 10x to 30x", self.Fore.RED + self.Style.BRIGHT)
    if LEARN_SCRIPT_AVAILABLE:
        self.print_color("🧠 SELF-LEARNING AI: ENABLED", self.Fore.MAGENTA + self.Style.BRIGHT)
    
    self.cycle_count = 0
    while True:
        try:
            self.cycle_count += 1
            self.print_color(f"\n🔄 TRADING CYCLE {self.cycle_count} (3MIN INTERVAL)", self.Fore.CYAN + self.Style.BRIGHT)
            self.print_color("=" * 60, self.Fore.CYAN)
            self.run_trading_cycle()
            self.print_color(f"⏳ Next analysis in 3 minute...", self.Fore.BLUE)
            time.sleep(self.monitoring_interval)  # 3 minute
            
        except KeyboardInterrupt:
            self.print_color(f"\n🛑 TRADING STOPPED", self.Fore.RED + self.Style.BRIGHT)
            self.show_trade_history(15)
            self.show_trading_stats()
            break
        except Exception as e:
            self.print_color(f"Main loop error: {e}", self.Fore.RED)
            time.sleep(self.monitoring_interval)

# Add remaining methods
remaining_methods = [
    get_ai_close_decision, monitor_positions, display_dashboard,
    show_trade_history, show_trading_stats, run_trading_cycle, start_trading
]

for method in remaining_methods:
    setattr(FullyAutonomous1HourAITrader, method.__name__, method)

# Paper trading class (simplified for brevity)
class FullyAutonomous1HourPaperTrader:
    def __init__(self, real_bot):
        self.real_bot = real_bot
        # Copy colorama attributes from real_bot
        self.Fore = real_bot.Fore
        self.Back = real_bot.Back
        self.Style = real_bot.Style
        self.COLORAMA_AVAILABLE = real_bot.COLORAMA_AVAILABLE
        
        # Copy reverse position settings
        self.allow_reverse_positions = True
        
        # NEW: Monitoring interval (3 minute)
        self.monitoring_interval = 180  # 3 minute in seconds
        
        self.paper_balance = 500  # Virtual $500 budget
        self.available_budget = 500
        self.paper_positions = {}
        self.paper_history_file = "fully_autonomous_1hour_paper_trading_history.json"
        self.paper_history = self.load_paper_history()
        self.available_pairs = ["BTCUSDT", "BNBUSDT", "SOLUSDT", "AVAXUSDT"]
        self.max_concurrent_trades = 6
        
        self.real_bot.print_color("🤖 FULLY AUTONOMOUS PAPER TRADER INITIALIZED!", self.Fore.GREEN + self.Style.BRIGHT)
        self.real_bot.print_color(f"💰 Virtual Budget: ${self.paper_balance}", self.Fore.CYAN + self.Style.BRIGHT)
        self.real_bot.print_color(f"🔄 REVERSE POSITION FEATURE: ENABLED", self.Fore.MAGENTA + self.Style.BRIGHT)
        self.real_bot.print_color(f"🎯 NO TP/SL - AI MANUAL CLOSE ONLY", self.Fore.YELLOW + self.Style.BRIGHT)
        self.real_bot.print_color(f"⏰ MONITORING: 1 MINUTE INTERVAL", self.Fore.RED + self.Style.BRIGHT)
        
    def load_paper_history(self):
        """Load PAPER trading history"""
        try:
            if os.path.exists(self.paper_history_file):
                with open(self.paper_history_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            self.real_bot.print_color(f"Error loading paper trade history: {e}", self.Fore.RED)
            return []
    
    def save_paper_history(self):
        """Save PAPER trading history"""
        try:
            with open(self.paper_history_file, 'w') as f:
                json.dump(self.paper_history, f, indent=2)
        except Exception as e:
            self.real_bot.print_color(f"Error saving paper trade history: {e}", self.Fore.RED)
    
    def add_paper_trade_to_history(self, trade_data):
        """Add trade to PAPER trading history"""
        try:
            trade_data['close_time'] = self.real_bot.get_thailand_time()
            trade_data['close_timestamp'] = time.time()
            trade_data['trade_type'] = 'PAPER'
            self.paper_history.append(trade_data)
            
            if len(self.paper_history) > 200:
                self.paper_history = self.paper_history[-200:]
            self.save_paper_history()
            self.real_bot.print_color(f"📝 PAPER Trade saved: {trade_data['pair']} {trade_data['direction']} P&L: ${trade_data.get('pnl', 0):.2f}", self.Fore.CYAN)
        except Exception as e:
            self.real_bot.print_color(f"Error adding paper trade to history: {e}", self.Fore.RED)

    def calculate_current_pnl(self, trade, current_price):
        """Calculate current PnL percentage for paper trading"""
        try:
            if trade['direction'] == 'LONG':
                pnl_percent = ((current_price - trade['entry_price']) / trade['entry_price']) * 100 * trade['leverage']
            else:
                pnl_percent = ((trade['entry_price'] - current_price) / trade['entry_price']) * 100 * trade['leverage']
            return pnl_percent
        except:
            return 0

    def paper_execute_reverse_position(self, pair, ai_decision, current_trade):
        """Execute reverse position in paper trading - CLOSE CURRENT, THEN ASK AI BEFORE OPENING REVERSE"""
        try:
            self.real_bot.print_color(f"🔄 PAPER: ATTEMPTING REVERSE POSITION FOR {pair}", self.Fore.YELLOW + self.Style.BRIGHT)
            
            # 1. First close the current losing position
            close_success = self.paper_close_trade_immediately(pair, current_trade, "REVERSE_POSITION")
            
            if close_success:
                # 2. Wait a moment and verify position is actually closed
                time.sleep(1)
                
                # Verify position is actually removed
                if pair in self.paper_positions:
                    self.real_bot.print_color(f"⚠️  PAPER: Position still exists after close, forcing removal...", self.Fore.RED)
                    del self.paper_positions[pair]
                
                # 3. 🆕 ASK AI AGAIN BEFORE OPENING REVERSE POSITION
                self.real_bot.print_color(f"🔍 PAPER: Asking AI to confirm reverse position for {pair}...", self.Fore.BLUE)
                market_data = self.real_bot.get_price_history(pair)
                
                # Get fresh AI decision after closing
                new_ai_decision = self.real_bot.get_ai_trading_decision(pair, market_data, None)
                
                # Check if AI still wants to open reverse position
                if new_ai_decision["decision"] in ["LONG", "SHORT"] and new_ai_decision["position_size_usd"] > 0:
                    # 🎯 Calculate correct reverse direction
                    current_direction = current_trade['direction']
                    if current_direction == "LONG":
                        correct_reverse_direction = "SHORT"
                    else:
                        correct_reverse_direction = "LONG"
                    
                    self.real_bot.print_color(f"✅ PAPER AI CONFIRMED: Opening {correct_reverse_direction} {pair}", self.Fore.CYAN + self.Style.BRIGHT)
                    
                    # Use the new AI decision but ensure correct direction
                    reverse_decision = new_ai_decision.copy()
                    reverse_decision["decision"] = correct_reverse_direction
                    
                    # Execute the reverse trade
                    return self.paper_execute_trade(pair, reverse_decision)
                else:
                    self.real_bot.print_color(f"🔄 PAPER AI changed mind, not opening reverse position for {pair}", self.Fore.YELLOW)
                    self.real_bot.print_color(f"📝 PAPER AI Decision: {new_ai_decision['decision']} | Reason: {new_ai_decision['reasoning']}", self.Fore.WHITE)
                    return False
            else:
                self.real_bot.print_color(f"❌ PAPER: Reverse position failed", self.Fore.RED)
                return False
                
        except Exception as e:
            self.real_bot.print_color(f"❌ PAPER: Reverse position execution failed: {e}", self.Fore.RED)
            return False

    def paper_close_trade_immediately(self, pair, trade, reason="REVERSE"):
        """Close paper trade immediately"""
        try:
            current_price = self.real_bot.get_current_price(pair)
            if trade['direction'] == 'LONG':
                pnl = (current_price - trade['entry_price']) * trade['quantity']
            else:
                pnl = (trade['entry_price'] - current_price) * trade['quantity']
            
            trade['status'] = 'CLOSED'
            trade['exit_price'] = current_price
            trade['pnl'] = pnl
            trade['close_reason'] = reason
            trade['close_time'] = self.real_bot.get_thailand_time()
            
            self.available_budget += trade['position_size_usd'] + pnl
            self.paper_balance = self.available_budget
            
            self.add_paper_trade_to_history(trade.copy())
            self.real_bot.print_color(f"✅ PAPER: Position closed for reverse: {pair} | P&L: ${pnl:.2f}", self.Fore.CYAN)
            
            # Remove from active positions after closing
            if pair in self.paper_positions:
                del self.paper_positions[pair]
            
            return True
                
        except Exception as e:
            self.real_bot.print_color(f"❌ PAPER: Immediate close failed: {e}", self.Fore.RED)
            return False

    def get_ai_close_decision(self, pair, trade):
        """Ask AI whether to close paper position"""
        try:
            current_price = self.real_bot.get_current_price(pair)
            market_data = self.real_bot.get_price_history(pair)
            current_pnl = self.calculate_current_pnl(trade, current_price)
            
            prompt = f"""
            SHOULD WE CLOSE THIS PAPER TRADING POSITION? (3MINUTE MONITORING)
            
            CURRENT ACTIVE PAPER TRADE:
            - Pair: {pair}
            - Direction: {trade['direction']}
            - Entry Price: ${trade['entry_price']:.4f}
            - Current Price: ${current_price:.4f}
            - PnL: {current_pnl:.2f}%
            - Position Size: ${trade['position_size_usd']:.2f}
            - Leverage: {trade['leverage']}x
            - Trade Age: {(time.time() - trade['entry_time']) / 60:.1f} minutes
            
            MARKET CONDITIONS:
            - 1H Change: {market_data.get('price_change', 0):.2f}%
            - Support: {market_data.get('support_levels', [])}
            - Resistance: {market_data.get('resistance_levels', [])}
            
            Should we CLOSE this paper position now?
            
            Return JSON:
            {{
                "should_close": true/false,
                "close_reason": "TAKE_PROFIT" | "STOP_LOSS" | "TREND_REVERSAL" | "TIME_EXIT",
                "confidence": 0-100,
                "reasoning": "Detailed analysis"
            }}
            """
            
            headers = {
                "Authorization": f"Bearer {self.real_bot.openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com",
                "X-Title": "Fully Autonomous AI Paper Trader"
            }
            
            data = {
                "model": "deepseek/deepseek-chat-v3.1",
                "messages": [
                    {"role": "system", "content": "You are an AI paper trader monitoring active positions every 1 minute. Decide whether to close paper positions based on current market conditions and technical analysis."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 600
            }
            
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=45)
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['choices'][0]['message']['content'].strip()
                
                json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    close_decision = json.loads(json_str)
                    return close_decision
                    
            return {"should_close": False, "close_reason": "AI_UNAVAILABLE", "confidence": 0, "reasoning": "AI analysis failed"}
            
        except Exception as e:
            self.real_bot.print_color(f"PAPER: AI close decision error: {e}", self.Fore.RED)
            return {"should_close": False, "close_reason": "ERROR", "confidence": 0, "reasoning": f"Error: {e}"}

    def paper_execute_trade(self, pair, ai_decision):
        """Execute paper trade WITHOUT TP/SL orders"""
        try:
            decision = ai_decision["decision"]
            position_size_usd = ai_decision["position_size_usd"]
            entry_price = ai_decision["entry_price"]
            leverage = ai_decision["leverage"]
            confidence = ai_decision["confidence"]
            reasoning = ai_decision["reasoning"]
            
            # Handle reverse positions
            if decision.startswith('REVERSE_'):
                if pair in self.paper_positions:
                    current_trade = self.paper_positions[pair]
                    return self.paper_execute_reverse_position(pair, ai_decision, current_trade)
                else:
                    self.real_bot.print_color(f"❌ PAPER: Cannot reverse - No active position for {pair}", self.Fore.RED)
                    return False
            
            if decision == "HOLD" or position_size_usd <= 0:
                self.real_bot.print_color(f"🟡 PAPER: DeepSeek decides to HOLD {pair}", self.Fore.YELLOW)
                return False
            
            # Check if we can open position
            if pair in self.paper_positions:
                self.real_bot.print_color(f"🚫 PAPER: Cannot open {pair}: Position already exists", self.Fore.RED)
                return False
            
            if len(self.paper_positions) >= self.max_concurrent_trades:
                self.real_bot.print_color(f"🚫 PAPER: Cannot open {pair}: Max concurrent trades reached (6)", self.Fore.RED)
                return False
                
            if position_size_usd > self.available_budget:
                self.real_bot.print_color(f"🚫 PAPER: Cannot open {pair}: Insufficient budget", self.Fore.RED)
                return False
            
            # Calculate quantity
            notional_value = position_size_usd * leverage
            quantity = notional_value / entry_price
            quantity = round(quantity, 3)
            
            # Display AI trade decision (NO TP/SL)
            direction_color = self.Fore.GREEN + self.Style.BRIGHT if decision == 'LONG' else self.Fore.RED + self.Style.BRIGHT
            direction_icon = "🟢 LONG" if decision == 'LONG' else "🔴 SHORT"
            
            self.real_bot.print_color(f"\n🤖 PAPER TRADE EXECUTION (NO TP/SL)", self.Fore.CYAN + self.Style.BRIGHT)
            self.real_bot.print_color("=" * 80, self.Fore.CYAN)
            self.real_bot.print_color(f"{direction_icon} {pair}", direction_color)
            self.real_bot.print_color(f"POSITION SIZE: ${position_size_usd:.2f}", self.Fore.GREEN + self.Style.BRIGHT)
            self.real_bot.print_color(f"LEVERAGE: {leverage}x ⚡", self.Fore.RED + self.Style.BRIGHT)
            self.real_bot.print_color(f"ENTRY PRICE: ${entry_price:.4f}", self.Fore.WHITE)
            self.real_bot.print_color(f"QUANTITY: {quantity}", self.Fore.CYAN)
            self.real_bot.print_color(f"🎯 NO TP/SL SET - AI WILL CLOSE MANUALLY", self.Fore.YELLOW + self.Style.BRIGHT)
            self.real_bot.print_color(f"CONFIDENCE: {confidence}%", self.Fore.YELLOW + self.Style.BRIGHT)
            self.real_bot.print_color(f"REASONING: {reasoning}", self.Fore.WHITE)
            self.real_bot.print_color("=" * 80, self.Fore.CYAN)
            
            # Update budget and track trade
            self.available_budget -= position_size_usd
            
            self.paper_positions[pair] = {
                "pair": pair,
                "direction": decision,
                "entry_price": entry_price,
                "quantity": quantity,
                "position_size_usd": position_size_usd,
                "leverage": leverage,
                "entry_time": time.time(),
                "status": 'ACTIVE',
                'ai_confidence': confidence,
                'ai_reasoning': reasoning,
                'entry_time_th': self.real_bot.get_thailand_time(),
                'has_tp_sl': False  # Mark as no TP/SL
            }
            
            self.real_bot.print_color(f"✅ PAPER TRADE EXECUTED (NO TP/SL): {pair} {decision} | Leverage: {leverage}x", self.Fore.GREEN + self.Style.BRIGHT)
            return True
            
        except Exception as e:
            self.real_bot.print_color(f"❌ PAPER: Trade execution failed: {e}", self.Fore.RED)
            return False

    def monitor_paper_positions(self):
        """Monitor paper positions and ask AI when to close"""
        try:
            closed_positions = []
            for pair, trade in list(self.paper_positions.items()):
                if trade['status'] != 'ACTIVE':
                    continue
                
                # Ask AI whether to close this paper position
                if not trade.get('has_tp_sl', True):
                    self.real_bot.print_color(f"🔍 PAPER: Checking if AI wants to close {pair}...", self.Fore.BLUE)
                    close_decision = self.get_ai_close_decision(pair, trade)
                    
                    if close_decision.get("should_close", False):
                        close_reason = close_decision.get("close_reason", "AI_DECISION")
                        confidence = close_decision.get("confidence", 0)
                        reasoning = close_decision.get("reasoning", "No reason provided")
                        
                        self.real_bot.print_color(f"🎯 PAPER AI Decision: CLOSE {pair} (Confidence: {confidence}%)", self.Fore.YELLOW + self.Style.BRIGHT)
                        self.real_bot.print_color(f"📝 Reason: {reasoning}", self.Fore.WHITE)
                        
                        success = self.paper_close_trade_immediately(pair, trade, f"AI_CLOSE: {close_reason}")
                        if success:
                            closed_positions.append(pair)
                    else:
                        # Show AI's decision to hold
                        if close_decision.get('confidence', 0) > 0:
                            self.real_bot.print_color(f"🔍 PAPER AI wants to HOLD {pair} (Confidence: {close_decision.get('confidence', 0)}%)", self.Fore.GREEN)
                    
            return closed_positions
                    
        except Exception as e:
            self.real_bot.print_color(f"PAPER: Monitoring error: {e}", self.Fore.RED)
            return []

    def show_paper_trade_history(self, limit=15):
        """Show PAPER trading history"""
        if not self.paper_history:
            self.real_bot.print_color("No PAPER trade history found", self.Fore.YELLOW)
            return
        
        self.real_bot.print_color(f"\n📝 PAPER TRADING HISTORY (Last {min(limit, len(self.paper_history))} trades)", self.Fore.GREEN + self.Style.BRIGHT)
        self.real_bot.print_color("=" * 120, self.Fore.GREEN)
        
        recent_trades = self.paper_history[-limit:]
        for i, trade in enumerate(reversed(recent_trades)):
            pnl = trade.get('pnl', 0)
            pnl_color = self.Fore.GREEN + self.Style.BRIGHT if pnl > 0 else self.Fore.RED + self.Style.BRIGHT if pnl < 0 else self.Fore.YELLOW
            direction_icon = "🟢 LONG" if trade['direction'] == 'LONG' else "🔴 SHORT"
            position_size = trade.get('position_size_usd', 0)
            leverage = trade.get('leverage', 1)
            
            self.real_bot.print_color(f"{i+1:2d}. {direction_icon} {trade['pair']} | Size: ${position_size:.2f} | Leverage: {leverage}x | P&L: ${pnl:.2f}", pnl_color)
            self.real_bot.print_color(f"     Entry: ${trade.get('entry_price', 0):.4f} | Exit: ${trade.get('exit_price', 0):.4f} | {trade.get('close_reason', 'N/A')}", self.Fore.YELLOW)

    def get_paper_portfolio_status(self):
        """Show paper trading portfolio status"""
        total_trades = len(self.paper_history)
        winning_trades = len([t for t in self.paper_history if t.get('pnl', 0) > 0])
        total_pnl = sum(trade.get('pnl', 0) for trade in self.paper_history)
        
        self.real_bot.print_color(f"\n📊 PAPER TRADING PORTFOLIO", self.Fore.CYAN + self.Style.BRIGHT)
        self.real_bot.print_color("=" * 70, self.Fore.CYAN)
        self.real_bot.print_color(f"Active Positions: {len(self.paper_positions)}/6", self.Fore.WHITE)
        self.real_bot.print_color(f"Available Budget: ${self.available_budget:.2f}", self.Fore.WHITE + self.Style.BRIGHT)
        self.real_bot.print_color(f"Total Paper Trades: {total_trades}", self.Fore.WHITE)
        
        if total_trades > 0:
            win_rate = (winning_trades / total_trades) * 100
            self.real_bot.print_color(f"Paper Win Rate: {win_rate:.1f}%", self.Fore.GREEN + self.Style.BRIGHT if win_rate > 50 else self.Fore.YELLOW)
            self.real_bot.print_color(f"Total Paper P&L: ${total_pnl:.2f}", self.Fore.GREEN + self.Style.BRIGHT if total_pnl > 0 else self.Fore.RED + self.Style.BRIGHT)
            avg_trade = total_pnl / total_trades
            self.real_bot.print_color(f"Average Paper P&L: ${avg_trade:.2f}", self.Fore.WHITE)

    def display_paper_dashboard(self):
        """Display paper trading dashboard"""
        self.real_bot.print_color(f"\n🤖 PAPER TRADING DASHBOARD - {self.real_bot.get_thailand_time()}", self.Fore.GREEN + self.Style.BRIGHT)
        self.real_bot.print_color("=" * 90, self.Fore.GREEN)
        self.real_bot.print_color(f"🎯 MODE: NO TP/SL - AI MANUAL CLOSE ONLY", self.Fore.YELLOW + self.Style.BRIGHT)
        self.real_bot.print_color(f"⏰ MONITORING: 1 MINUTE INTERVAL", self.Fore.RED + self.Style.BRIGHT)
        
        active_count = 0
        total_unrealized = 0
        
        for pair, trade in self.paper_positions.items():
            if trade['status'] == 'ACTIVE':
                active_count += 1
                current_price = self.real_bot.get_current_price(pair)
                
                direction_icon = "🟢 LONG" if trade['direction'] == 'LONG' else "🔴 SHORT"
                
                if trade['direction'] == 'LONG':
                    unrealized_pnl = (current_price - trade['entry_price']) * trade['quantity']
                else:
                    unrealized_pnl = (trade['entry_price'] - current_price) * trade['quantity']
                    
                total_unrealized += unrealized_pnl
                pnl_color = self.Fore.GREEN + self.Style.BRIGHT if unrealized_pnl >= 0 else self.Fore.RED + self.Style.BRIGHT
                
                self.real_bot.print_color(f"{direction_icon} {pair}", self.Fore.WHITE + self.Style.BRIGHT)
                self.real_bot.print_color(f"   Size: ${trade['position_size_usd']:.2f} | Leverage: {trade['leverage']}x ⚡", self.Fore.WHITE)
                self.real_bot.print_color(f"   Entry: ${trade['entry_price']:.4f} | Current: ${current_price:.4f}", self.Fore.WHITE)
                self.real_bot.print_color(f"   P&L: ${unrealized_pnl:.2f}", pnl_color)
                self.real_bot.print_color(f"   🎯 NO TP/SL - AI Monitoring Every 3min", self.Fore.YELLOW)
                self.real_bot.print_color("   " + "-" * 60, self.Fore.GREEN)
        
        if active_count == 0:
            self.real_bot.print_color("No active paper positions", self.Fore.YELLOW)
        else:
            total_color = self.Fore.GREEN + self.Style.BRIGHT if total_unrealized >= 0 else self.Fore.RED + self.Style.BRIGHT
            self.real_bot.print_color(f"📊 Active Paper Positions: {active_count}/6 | Total Unrealized P&L: ${total_unrealized:.2f}", total_color)

    def run_paper_trading_cycle(self):
        """Run one complete paper trading cycle with REVERSE feature and AI manual close"""
        try:
            self.monitor_paper_positions()
            self.display_paper_dashboard()
            
            # Show paper history every 10 cycles (10 minutes)
            if hasattr(self, 'paper_cycle_count') and self.paper_cycle_count % 10 == 0:
                self.show_paper_trade_history(8)
            
            self.get_paper_portfolio_status()
            
            self.real_bot.print_color(f"\n🔍 DEEPSEEK SCANNING FOR PAPER TRADES WITH ${self.available_budget:.2f} AVAILABLE...", self.Fore.BLUE + self.Style.BRIGHT)
            
            qualified_signals = 0
            for pair in self.available_pairs:
                if self.available_budget > 100:
                    market_data = self.real_bot.get_price_history(pair)
                    
                    # Pass current trade to AI for reverse analysis
                    current_trade = self.paper_positions.get(pair)
                    ai_decision = self.real_bot.get_ai_trading_decision(pair, market_data, current_trade)
                    
                    if ai_decision["decision"] != "HOLD" and ai_decision["position_size_usd"] > 0:
                        qualified_signals += 1
                        direction = ai_decision['decision']
                        
                        if direction.startswith('REVERSE_'):
                            self.real_bot.print_color(f"🔄 PAPER REVERSE SIGNAL: {pair} {direction} | Size: ${ai_decision['position_size_usd']:.2f}", self.Fore.YELLOW + self.Style.BRIGHT)
                        else:
                            leverage_info = f"Leverage: {ai_decision['leverage']}x"
                            self.real_bot.print_color(f"🎯 PAPER TRADE SIGNAL: {pair} {direction} | Size: ${ai_decision['position_size_usd']:.2f} | {leverage_info}", self.Fore.GREEN + self.Style.BRIGHT)
                        
                        self.paper_execute_trade(pair, ai_decision)
                        time.sleep(1)  # Reduced delay for faster 3min cycles
                
            if qualified_signals > 0:
                self.real_bot.print_color(f"🎯 {qualified_signals} qualified paper signals executed", self.Fore.GREEN + self.Style.BRIGHT)
            else:
                self.real_bot.print_color("No qualified paper signals this cycle", self.Fore.YELLOW)
            
        except Exception as e:
            self.real_bot.print_color(f"PAPER: Trading cycle error: {e}", self.Fore.RED)

    def start_paper_trading(self):
        """Start paper trading with REVERSE feature and NO TP/SL"""
        self.real_bot.print_color("🚀 STARTING PAPER TRADING WITH 3MINUTE MONITORING!", self.Fore.GREEN + self.Style.BRIGHT)
        self.real_bot.print_color("💰 VIRTUAL $500 BUDGET - NO REAL MONEY", self.Fore.CYAN + self.Style.BRIGHT)
        self.real_bot.print_color("🔄 REVERSE POSITION: ENABLED (AI can flip losing positions)", self.Fore.MAGENTA + self.Style.BRIGHT)
        self.real_bot.print_color("🎯 NO TP/SL - AI MANUAL CLOSE ONLY", self.Fore.YELLOW + self.Style.BRIGHT)
        self.real_bot.print_color("⏰ MONITORING: 3 MINUTE INTERVAL", self.Fore.RED + self.Style.BRIGHT)
        
        self.paper_cycle_count = 0
        while True:
            try:
                self.paper_cycle_count += 1
                self.real_bot.print_color(f"\n🔄 PAPER TRADING CYCLE {self.paper_cycle_count} (3MIN INTERVAL)", self.Fore.GREEN)
                self.real_bot.print_color("=" * 60, self.Fore.GREEN)
                self.run_paper_trading_cycle()
                self.real_bot.print_color(f"⏳ DeepSeek analyzing next paper opportunities in 1 minute...", self.Fore.BLUE)
                time.sleep(self.monitoring_interval)
                
            except KeyboardInterrupt:
                self.real_bot.print_color(f"\n🛑 PAPER TRADING STOPPED", self.Fore.RED + self.Style.BRIGHT)
                
                # Show final paper trading results
                total_trades = len(self.paper_history)
                if total_trades > 0:
                    winning_trades = len([t for t in self.paper_history if t.get('pnl', 0) > 0])
                    total_pnl = sum(trade.get('pnl', 0) for trade in self.paper_history)
                    win_rate = (winning_trades / total_trades) * 100
                    
                    self.real_bot.print_color(f"\n📊 FINAL PAPER TRADING RESULTS", self.Fore.CYAN + self.Style.BRIGHT)
                    self.real_bot.print_color("=" * 50, self.Fore.CYAN)
                    self.real_bot.print_color(f"Total Paper Trades: {total_trades}", self.Fore.WHITE)
                    self.real_bot.print_color(f"Paper Win Rate: {win_rate:.1f}%", self.Fore.GREEN)
                    self.real_bot.print_color(f"Total Paper P&L: ${total_pnl:.2f}", self.Fore.GREEN if total_pnl > 0 else self.Fore.RED)
                    self.real_bot.print_color(f"Final Paper Balance: ${self.paper_balance:.2f}", self.Fore.CYAN + self.Style.BRIGHT)
                
                break
            except Exception as e:
                self.real_bot.print_color(f"PAPER: Trading error: {e}", self.Fore.RED)
                time.sleep(self.monitoring_interval)
                
if __name__ == "__main__":
    try:
        ai_trader = FullyAutonomous1HourAITrader()
        
        print("\n" + "="*80)
        print("🤖 AI TRADER WITH 3MINUTE MONITORING & ENHANCED REVERSE FEATURE")
        print("="*80)
        print("SELECT MODE:")
        print("1. 🚀 Live Trading (Real Money)")
        print("2. 💸 Paper Trading (Virtual Money)")
        
        choice = input("Enter choice (1-2): ").strip()
        
        if choice == "1":
            print("⚠️  WARNING: REAL MONEY TRADING! ⚠️")
            print("🔄 REVERSE POSITION FEATURE: ENABLED")
            print("🎯 NO TP/SL - AI MANUAL CLOSE ONLY")
            print("⏰ MONITORING: 3 MINUTE INTERVAL")
            print("⚡ AI CAN FLIP LOSING POSITIONS (WITH CONFIRMATION)")
            if LEARN_SCRIPT_AVAILABLE:
                print("🧠 SELF-LEARNING AI: ENABLED")
            confirm = input("Type '3MINUTE' to confirm: ").strip()
            if confirm.upper() == '3MINUTE':
                ai_trader.start_trading()
            else:
                print("Using Paper Trading mode instead...")
                paper_bot = FullyAutonomous1HourPaperTrader(ai_trader)
                paper_bot.start_paper_trading()
        else:
            paper_bot = FullyAutonomous1HourPaperTrader(ai_trader)
            paper_bot.start_paper_trading()
            
    except Exception as e:
        print(f"Failed to start AI trader: {e}")
