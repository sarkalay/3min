# 1_core.py
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from learn_script import SelfLearningAITrader
    LEARN_SCRIPT_AVAILABLE = True
    print("Learn script loaded successfully!")
except ImportError as e:
    print(f"Learn script import failed: {e}")
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
from datetime import datetime
import pytz
import pandas as pd

try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    print("Warning: Colorama not installed. Run: pip install colorama")

load_dotenv()

if not COLORAMA_AVAILABLE:
    class DummyColors:
        def __getattr__(self, name): return ''
    Fore = DummyColors()
    Back = DummyColors()
    Style = DummyColors()

# === CLASS DEFINITION ===
if LEARN_SCRIPT_AVAILABLE:
    class FullyAutonomous1HourAITrader(SelfLearningAITrader):
        def __init__(self):
            super().__init__()
            self._initialize_trading()
else:
    class FullyAutonomous1HourAITrader(object):
        def __init__(self):
            self.mistakes_history = []
            self.learned_patterns = {}
            self.performance_stats = {
                'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
                'common_mistakes': {}, 'improvement_areas': []
            }
            self._initialize_trading()

# === COMMON INITIALIZATION ===
def _initialize_trading(self):
    self.binance_api_key = os.getenv('BINANCE_API_KEY')
    self.binance_secret = os.getenv('BINANCE_SECRET_KEY')
    self.openrouter_key = os.getenv('OPENROUTER_API_KEY')

    self.Fore = Fore
    self.Back = Back
    self.Style = Style
    self.COLORAMA_AVAILABLE = COLORAMA_AVAILABLE
    self.thailand_tz = pytz.timezone('Asia/Bangkok')

    self.total_budget = 500
    self.available_budget = 500
    self.max_position_size_percent = 10
    self.max_concurrent_trades = 4
    self.available_pairs = ["BNBUSDT", "SOLUSDT", "AVAXUSDT"]
    self.ai_opened_trades = {}
    self.real_trade_history_file = "fully_autonomous_1hour_ai_trading_history.json"
    self.real_trade_history = self.load_real_trade_history()
    self.real_total_trades = 0
    self.real_winning_trades = 0
    self.real_total_pnl = 0.0
    self.quantity_precision = {}
    self.price_precision = {}
    self.allow_reverse_positions = True
    self.monitoring_interval = 180  # 3 minutes

    self.validate_api_keys()
    try:
        self.binance = Client(self.binance_api_key, self.binance_secret)
        self.print_color("FULLY AUTONOMOUS AI TRADER ACTIVATED!", self.Fore.CYAN + self.Style.BRIGHT)
        self.print_color(f"TOTAL BUDGET: ${self.total_budget}", self.Fore.GREEN + self.Style.BRIGHT)
        self.print_color("REVERSE POSITION: ENABLED", self.Fore.MAGENTA + self.Style.BRIGHT)
        self.print_color("NO TP/SL - AI MANUAL CLOSE", self.Fore.YELLOW + self.Style.BRIGHT)
        self.print_color("MONITORING: 3 MINUTE INTERVAL", self.Fore.RED + self.Style.BRIGHT)
    except Exception as e:
        self.print_color(f"Binance init failed: {e}", self.Fore.RED)
        self.binance = None

    self.validate_config()
    if self.binance:
        self.setup_futures()
        self.load_symbol_precision()

# Attach
FullyAutonomous1HourAITrader._initialize_trading = _initialize_trading
