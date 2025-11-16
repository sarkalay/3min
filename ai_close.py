# 4_ai_close.py
import requests
import re
import json
import time

def get_ai_close_decision(self, pair, trade):
    try:
        current_price = self.get_current_price(pair)
        market_data = self.get_price_history(pair)
        pnl = self.calculate_current_pnl(trade, current_price)

        prompt = f"""
        SHOULD WE CLOSE THIS POSITION? (3MIN MONITORING)
        Pair: {pair} | Dir: {trade['direction']} | Entry: ${trade['entry_price']:.4f}
        Current: ${current_price:.4f} | PnL: {pnl:.2f}% | Age: {(time.time()-trade['entry_time'])/60:.1f}min
        1H Change: {market_data.get('price_change',0):.2f}%

        First, think step-by-step about the market. Then return JSON:
        {{
            "should_close": true/false,
            "close_reason": "TAKE_PROFIT" | "STOP_LOSS" | "TREND_REVERSAL" | "TIME_EXIT",
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
            "max_tokens": 600
        }

        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=45)
        if response.status_code == 200:
            ai_response = response.json()['choices'][0]['message']['content'].strip()
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group())
                thought = ai_response[:json_match.start()].strip()
                if not thought:
                    thought = "No detailed explanation."
                decision['ai_thought'] = thought
                return decision
        return {"should_close": False, "close_reason": "API_ERROR", "ai_thought": "Failed to get response"}
    except Exception as e:
        return {"should_close": False, "close_reason": "EXCEPTION", "ai_thought": str(e)}

def monitor_positions(self):
    closed = []
    for pair, trade in list(self.ai_opened_trades.items()):
        if trade['status'] != 'ACTIVE': continue
        decision = self.get_ai_close_decision(pair, trade)
        if decision.get("should_close"):
            thought = decision.get("ai_thought", "")
            reason = f"AI_CLOSE: {decision['close_reason']} | {thought[:100]}{'...' if len(thought)>100 else ''}"
            self.print_color(f"AI Decision: CLOSE {pair}", self.Fore.YELLOW + self.Style.BRIGHT)
            self.print_color(f"AI Thought: {thought}", self.Fore.CYAN + self.Style.DIM)
            self.close_trade_immediately(pair, trade, reason)
            closed.append(pair)
    return closed

# Attach
for func in [get_ai_close_decision, monitor_positions]:
    setattr(FullyAutonomous1HourAITrader, func.__name__, func)
