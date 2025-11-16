# 4_ai_close.py
def get_ai_close_decision(self, pair, trade):
    try:
        current_price = self.get_current_price(pair)
        market_data = self.get_price_history(pair)
        pnl = self.calculate_current_pnl(trade, current_price)

        prompt = f"""
        SHOULD CLOSE? Pair: {pair} | Dir: {trade['direction']} | Entry: ${trade['entry_price']:.4f}
        Current: ${current_price:.4f} | PnL: {pnl:.2f}% | Age: {(time.time()-trade['entry_time'])/60:.1f}min
        1H Change: {market_data.get('price_change',0):.2f}%
        First, think step-by-step. Then return JSON:
        {{
            "should_close": true/false,
            "close_reason": "TAKE_PROFIT" | "STOP_LOSS" | "TREND_REVERSAL",
            "confidence": 0-100,
            "reasoning": "Detailed reason"
        }}
        """

        # ... API call
        response = requests.post(...)
        if response.status_code == 200:
            ai_response = response.json()['choices'][0]['message']['content'].strip()
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group())
                decision['ai_thought'] = ai_response[:json_match.start()].strip()
                return decision
    except Exception as e:
        return {"should_close": False, "ai_thought": str(e)}

def monitor_positions(self):
    for pair, trade in list(self.ai_opened_trades.items()):
        if trade['status'] != 'ACTIVE': continue
        close_decision = self.get_ai_close_decision(pair, trade)
        if close_decision.get("should_close"):
            thought = close_decision.get("ai_thought", "")
            reason = f"AI_CLOSE: {close_decision['close_reason']} | {thought[:80]}..."
            self.close_trade_immediately(pair, trade, reason)
            self.print_color(f"AI Thought: {thought}", self.Fore.CYAN + self.Style.DIM)

# Attach
for func in [get_ai_close_decision, monitor_positions]:
    setattr(FullyAutonomous1HourAITrader, func.__name__, func)
