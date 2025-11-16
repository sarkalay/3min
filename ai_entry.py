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
            mtf_text = "".join([
                f"- {tf.upper()}: {d.get('trend')} | RSI: {d.get('rsi')} | Vol: {'SPIKE' if d.get('vol_spike') else 'Normal'}\n"
                for tf, d in mtf.items()
            ])

            h1_trend = mtf.get('1h', {}).get('trend')
            h4_trend = mtf.get('4h', {}).get('trend')
            alignment = "STRONG" if h1_trend == h4_trend and h1_trend else "WEAK"

            reverse_analysis = ""
            if current_trade and self.allow_reverse_positions:
                pnl = self.calculate_current_pnl(current_trade, current_price)
                reverse_analysis = f"Current PnL: {pnl:.2f}% | Direction: {current_trade['direction']}"

            learning_context = ""
            if LEARN_SCRIPT_AVAILABLE and hasattr(self, 'get_learning_enhanced_prompt'):
                learning_context = self.get_learning_enhanced_prompt(pair, market_data)

            prompt = f"""
            YOU ARE A PROFESSIONAL AI TRADER. Budget: ${self.available_budget:.2f}
            {mtf_text}
            TREND ALIGNMENT: {alignment}
            Pair: {pair} | Price: ${current_price:.6f}
            {reverse_analysis}
            {learning_context}

            RULES: Only trade if 1H+4H align. Confirm with 15m crossover + volume.
            Position: 5-10% budget. Leverage: 5-10x.
            REVERSE only if PnL ≤ -2% + trend flip + volume spike.

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
                "Content-Type": "application/json"
            }
            data = {
                "model": "deepseek/deepseek-chat-v3.1",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 800
            }

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
    except: pass
    return self.get_improved_fallback_decision(pair, {'current_price': current_price})

def get_improved_fallback_decision(self, pair, market_data):
    # ... (same as before)
    pass

def execute_ai_trade(self, pair, ai_decision):
    # ... (same as before)
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
