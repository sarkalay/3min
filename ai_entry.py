# 3_ai_entry.py
import time
import re
import json

def get_ai_trading_decision(self, pair, market_data, current_trade=None):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if not self.openrouter_key:
                return self.get_improved_fallback_decision(pair, market_data)

            current_price = market_data['current_price']
            mtf = market_data.get('mtf_analysis', {})
            mtf_text = ""
            for tf, d in mtf.items():
                mtf_text += f"- {tf.upper()}: {d.get('trend')} | RSI: {d.get('rsi')} | Vol: {'SPIKE' if d.get('vol_spike') else 'Normal'} | S/R: {d.get('support',0):.4f}/{d.get('resistance',0):.4f}\n"

            h1_trend = mtf.get('1h', {}).get('trend')
            h4_trend = mtf.get('4h', {}).get('trend')
            alignment = "STRONG" if h1_trend == h4_trend and h1_trend else "WEAK"

            reverse_analysis = ""
            if current_trade and self.allow_reverse_positions:
                pnl = self.calculate_current_pnl(current_trade, current_price)
                reverse_analysis = f"Current PnL: {pnl:.2f}% | Direction: {current_trade['direction']}"

            prompt = f"""
            YOU ARE A PROFESSIONAL AI TRADER. Budget: ${self.available_budget:.2f}
            {mtf_text}
            TREND ALIGNMENT: {alignment}
            Pair: {pair} | Price: ${current_price:.6f}
            {reverse_analysis}

            RULES:
            - Only trade if 1H and 4H trend align
            - Confirm entry with 15m crossover + volume spike
            - RSI < 30 = oversold, > 70 = overbought
            - Position size: 5-10% of budget
            - Leverage: 5-10x
            - NO TP/SL

            REVERSE only if:
            1. PnL ≤ -2%
            2. 1H+4H trend flipped
            3. 15m crossover in new direction
            4. Volume spike

            Return JSON:
            {{
                "decision": "LONG" | "SHORT" | "HOLD" | "REVERSE_LONG" | "REVERSE_SHORT",
                "position_size_usd": number,
                "entry_price": number,
                "leverage": number,
                "confidence": 0-100,
                "reasoning": "Brief reason"
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
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 800
            }

            self.print_color(f"Analyzing {pair}...", self.Fore.MAGENTA + self.Style.BRIGHT)
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=60)
            if response.status_code == 200:
                ai_response = response.json()['choices'][0]['message']['content'].strip()
                return self.parse_ai_trading_decision(ai_response, pair, current_price, current_trade)
        except Exception as e:
            if attempt == max_retries - 1:
                return self.get_improved_fallback_decision(pair, market_data)
            time.sleep(2)
    return self.get_improved_fallback_decision(pair, market_data)

def parse_ai_trading_decision(self, ai_response, pair, current_price, current_trade=None):
    try:
        json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            decision = data.get('decision', 'HOLD').upper()
            return {
                "decision": decision,
                "position_size_usd": float(data.get('position_size_usd', 0)),
                "entry_price": float(data.get('entry_price', current_price)),
                "leverage": max(5, min(10, int(data.get('leverage', 5)))),
                "confidence": float(data.get('confidence', 50)),
                "reasoning": data.get('reasoning', 'AI Analysis'),
                "should_reverse": decision.startswith('REVERSE_')
            }
    except Exception as e:
        self.print_color(f"Parse failed: {e}", self.Fore.RED)
    return self.get_improved_fallback_decision(pair, {'current_price': current_price})

def get_improved_fallback_decision(self, pair, market_data):
    current_price = market_data['current_price']
    mtf = market_data.get('mtf_analysis', {})
    h1 = mtf.get('1h', {})
    h4 = mtf.get('4h', {})
    m15 = mtf.get('15m', {})

    bullish = 0
    bearish = 0
    if h1.get('trend') == 'BULLISH': bullish += 1
    if h1.get('trend') == 'BEARISH': bearish += 1
    if h4.get('trend') == 'BULLISH': bullish += 1
    if h4.get('trend') == 'BEARISH': bearish += 1
    if h1.get('rsi', 50) < 35: bullish += 1
    if h1.get('rsi', 50) > 65: bearish += 1
    if m15.get('crossover') == 'GOLDEN': bullish += 1
    if m15.get('crossover') == 'DEATH': bearish += 1

    if bullish >= 3 and bearish <= 1:
        return {"decision": "LONG", "position_size_usd": 20, "entry_price": current_price, "leverage": 5, "confidence": 60, "reasoning": "Fallback Bullish", "should_reverse": False}
    elif bearish >= 3 and bullish <= 1:
        return {"decision": "SHORT", "position_size_usd": 20, "entry_price": current_price, "leverage": 5, "confidence": 60, "reasoning": "Fallback Bearish", "should_reverse": False}
    else:
        return {"decision": "HOLD", "position_size_usd": 0, "entry_price": current_price, "leverage": 5, "confidence": 40, "reasoning": "Mixed signals", "should_reverse": False}

def execute_ai_trade(self, pair, ai_decision):
    # ... (same as original, no changes needed here)
    pass

def calculate_current_pnl(self, trade, current_price):
    try:
        if trade['direction'] == 'LONG':
            return ((current_price - trade['entry_price']) / trade['entry_price']) * 100 * trade['leverage']
        else:
            return ((trade['entry_price'] - current_price) / trade['entry_price']) * 100 * trade['leverage']
    except:
        return 0

# Attach
for func in [get_ai_trading_decision, parse_ai_trading_decision, get_improved_fallback_decision, execute_ai_trade, calculate_current_pnl]:
    setattr(FullyAutonomous1HourAITrader, func.__name__, func)
