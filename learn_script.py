import os
import json
import time
from datetime import datetime

class SelfLearningAITrader:
    def __init__(self):
        # Mistake history storage
        self.mistakes_history_file = "ai_trading_mistakes.json"
        self.mistakes_history = self.load_mistakes_history()
        
        # Learning patterns
        self.learned_patterns = {}
        
        # Performance tracking
        self.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'common_mistakes': {},
            'improvement_areas': []
        }
    
    def load_mistakes_history(self):
        """Load AI's mistake history"""
        try:
            if os.path.exists(self.mistakes_history_file):
                with open(self.mistakes_history_file, 'r') as f:
                    return json.load(f)
            return []
        except:
            return []
    
    def save_mistakes_history(self):
        """Save mistake history"""
        try:
            with open(self.mistakes_history_file, 'w') as f:
                json.dump(self.mistakes_history, f, indent=2)
        except Exception as e:
            self.print_color(f"Error saving mistakes history: {e}", self.Fore.RED)
    
    def analyze_trade_mistake(self, trade):
        """Analyze what went wrong in a losing trade"""
        try:
            pnl = trade.get('pnl', 0)
            if pnl >= 0:
                return None  # Not a mistake if profitable
            
            mistake_analysis = {
                'timestamp': time.time(),
                'trade_data': trade,
                'mistake_type': self.identify_mistake_type(trade),
                'lesson_learned': self.extract_lesson(trade),
                'avoidance_strategy': self.create_avoidance_strategy(trade)
            }
            
            return mistake_analysis
        except Exception as e:
            self.print_color(f"Mistake analysis error: {e}", self.Fore.RED)
            return None
    
    def identify_mistake_type(self, trade):
        """Identify the type of mistake made"""
        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)
        direction = trade.get('direction', '')
        reasoning = trade.get('ai_reasoning', '').lower()
        
        # Common mistake patterns
        if "reverse" in trade.get('close_reason', '').lower():
            return "PREMATURE_REVERSAL"
        
        if abs(entry_price - exit_price) / entry_price > 0.1:  # 10% move against
            return "IGNORED_TREND"
        
        if "overbought" in reasoning and direction == "LONG":
            return "CONTRADICTORY_SIGNAL"
        
        if "oversold" in reasoning and direction == "SHORT":
            return "CONTRADICTORY_SIGNAL"
            
        return "STRATEGY_ERROR"
    
    def extract_lesson(self, trade):
        """Extract specific lesson from mistake"""
        mistake_type = self.identify_mistake_type(trade)
        direction = trade.get('direction', '')
        pnl = trade.get('pnl', 0)
        
        lessons = {
            "PREMATURE_REVERSAL": f"Avoid reversing positions too quickly. {direction} position lost ${abs(pnl):.2f}",
            "IGNORED_TREND": "Respect the stronger trend direction. Don't fight momentum",
            "CONTRADICTORY_SIGNAL": f"Don't take {direction} positions when indicators show opposite signals",
            "STRATEGY_ERROR": "Review entry/exit logic for this market condition"
        }
        
        return lessons.get(mistake_type, "Need better risk management")
    
    def create_avoidance_strategy(self, trade):
        """Create strategy to avoid repeating this mistake"""
        mistake_type = self.identify_mistake_type(trade)
        
        strategies = {
            "PREMATURE_REVERSAL": "Wait for stronger confirmation before reversing. Minimum 3% price confirmation",
            "IGNORED_TREND": "Check higher timeframe trend. Only trade with trend direction",
            "CONTRADICTORY_SIGNAL": "Require all indicators to align before entry",
            "STRATEGY_ERROR": "Reduce position size in uncertain market conditions"
        }
        
        return strategies.get(mistake_type, "Increase analysis confidence threshold")
    
    def learn_from_mistake(self, trade):
        """Main function to learn from mistakes"""
        mistake_analysis = self.analyze_trade_mistake(trade)
        
        if mistake_analysis:
            # Add to mistake history
            self.mistakes_history.append(mistake_analysis)
            
            # Update performance stats
            self.performance_stats['losing_trades'] += 1
            mistake_type = mistake_analysis['mistake_type']
            self.performance_stats['common_mistakes'][mistake_type] = \
                self.performance_stats['common_mistakes'].get(mistake_type, 0) + 1
            
            # Save the learning
            self.save_mistakes_history()
            self.update_learned_patterns(mistake_analysis)
            
            self.print_color(f"🧠 AI LEARNING: {mistake_analysis['lesson_learned']}", self.Fore.YELLOW)
            self.print_color(f"🛡️  AVOIDANCE: {mistake_analysis['avoidance_strategy']}", self.Fore.CYAN)
    
    def update_learned_patterns(self, mistake_analysis):
        """Update AI's learned patterns based on mistakes"""
        mistake_type = mistake_analysis['mistake_type']
        
        if mistake_type not in self.learned_patterns:
            self.learned_patterns[mistake_type] = {
                'occurrences': 0,
                'total_loss': 0,
                'avoidance_strategies': [],
                'last_occurrence': None
            }
        
        self.learned_patterns[mistake_type]['occurrences'] += 1
        self.learned_patterns[mistake_type]['total_loss'] += abs(mistake_analysis['trade_data'].get('pnl', 0))
        self.learned_patterns[mistake_type]['last_occurrence'] = time.time()
        
        # Add new avoidance strategy if different
        new_strategy = mistake_analysis['avoidance_strategy']
        if new_strategy not in self.learned_patterns[mistake_type]['avoidance_strategies']:
            self.learned_patterns[mistake_type]['avoidance_strategies'].append(new_strategy)
    
    def get_learning_enhanced_prompt(self, pair, market_data):
        """Create prompt enhanced with learned lessons"""
        
        # Get recent mistakes to avoid
        recent_mistakes = self.mistakes_history[-5:]  # Last 5 mistakes
        common_mistakes = self.get_most_common_mistakes()
        
        learning_section = """
        IMPORTANT LESSONS FROM MY RECENT MISTAKES:
        
        RECENT ERRORS TO AVOID:
        """
        
        for mistake in recent_mistakes:
            learning_section += f"- {mistake['lesson_learned']}\n"
            learning_section += f"  STRATEGY: {mistake['avoidance_strategy']}\n"
        
        learning_section += f"""
        
        MY MOST COMMON MISTAKES:
        """
        
        for mistake_type, count in common_mistakes.items():
            learning_section += f"- {mistake_type}: {count} times\n"
        
        learning_section += """
        
        APPLY THESE LESSONS TO CURRENT TRADE:
        - Double-check for patterns that caused previous losses
        - Ensure I'm not repeating known mistakes
        - Use stricter confirmation for error-prone setups
        """
        
        return learning_section
    
    def get_most_common_mistakes(self):
        """Get most frequently occurring mistakes"""
        return dict(sorted(
            self.performance_stats['common_mistakes'].items(),
            key=lambda x: x[1], 
            reverse=True
        )[:3])  # Top 3 mistakes
    
    def should_avoid_trade(self, ai_decision, market_data):
        """Check if this trade matches patterns that caused previous losses"""
        
        decision = ai_decision["decision"]
        reasoning = ai_decision["reasoning"].lower()
        
        # Check against learned mistake patterns
        for mistake_type, pattern_data in self.learned_patterns.items():
            if self.matches_mistake_pattern(ai_decision, market_data, mistake_type):
                self.print_color(f"🚫 BLOCKED: This matches previous '{mistake_type}' error pattern", self.Fore.RED)
                return True
        
        return False
    
    def matches_mistake_pattern(self, ai_decision, market_data, mistake_type):
        """Check if current setup matches a known mistake pattern"""
        
        if mistake_type == "PREMATURE_REVERSAL":
            # Check if reversing too quickly
            current_price = market_data['current_price']
            price_change = market_data.get('price_change', 0)
            
            if "reverse" in ai_decision.get('reasoning', '').lower():
                if abs(price_change) < 2:  # Less than 2% move
                    return True
        
        elif mistake_type == "IGNORED_TREND":
            # Check if going against strong trend
            price_change = market_data.get('price_change', 0)
            decision = ai_decision["decision"]
            
            if (price_change > 3 and decision == "SHORT") or (price_change < -3 and decision == "LONG"):
                return True
        
        elif mistake_type == "CONTRADICTORY_SIGNAL":
            # Check for conflicting signals in reasoning
            reasoning = ai_decision.get('reasoning', '').lower()
            decision = ai_decision["decision"]
            
            bearish_terms = ["down", "drop", "fall", "bear", "resistance", "overbought"]
            bullish_terms = ["up", "rise", "bull", "support", "oversold"]
            
            if decision == "LONG" and any(term in reasoning for term in bearish_terms):
                return True
            if decision == "SHORT" and any(term in reasoning for term in bullish_terms):
                return True
        
        return False

    def get_ai_decision_with_learning(self, pair, market_data):
        """Get AI decision enhanced with learned lessons"""
        
        # First get normal AI decision
        ai_decision = self.get_ai_trading_decision(pair, market_data)
        
        # Check if this matches known mistake patterns
        if self.should_avoid_trade(ai_decision, market_data):
            self.print_color(f"🧠 AI USING LEARNING: Blocking potential mistake for {pair}", self.Fore.YELLOW)
            return {
                "decision": "HOLD",
                "position_size_usd": 0,
                "entry_price": market_data['current_price'],
                "leverage": 10,
                "confidence": 0,
                "reasoning": f"Blocked - matches known error pattern. Learning from {len(self.mistakes_history)} past mistakes",
                "should_reverse": False
            }
        
        # Add learning context to reasoning
        if ai_decision["decision"] != "HOLD":
            learning_context = f" | Applying lessons from {len(self.mistakes_history)} past mistakes"
            ai_decision["reasoning"] += learning_context
        
        return ai_decision

    def show_learning_progress(self):
        """Display AI's learning progress"""
        total_mistakes = len(self.mistakes_history)
        common_mistakes = self.get_most_common_mistakes()
        
        self.print_color(f"\n🧠 AI LEARNING PROGRESS REPORT", self.Fore.CYAN + self.Style.BRIGHT)
        self.print_color("=" * 60, self.Fore.CYAN)
        self.print_color(f"Total Mistakes Learned From: {total_mistakes}", self.Fore.WHITE)
        self.print_color(f"Total Trades: {self.performance_stats['total_trades']}", self.Fore.WHITE)
        
        if total_mistakes > 0:
            mistake_rate = (self.performance_stats['losing_trades'] / self.performance_stats['total_trades']) * 100
            self.print_color(f"Mistake Rate: {mistake_rate:.1f}%", 
                           self.Fore.GREEN if mistake_rate < 30 else self.Fore.YELLOW)
        
        self.print_color(f"\n📊 MOST COMMON MISTAKES:", self.Fore.YELLOW)
        for mistake, count in common_mistakes.items():
            self.print_color(f"  {mistake}: {count} occurrences", self.Fore.WHITE)
        
        self.print_color(f"\n🛡️  ACTIVE PROTECTIONS:", self.Fore.GREEN)
        for mistake_type in common_mistakes.keys():
            self.print_color(f"  ✓ Blocking {mistake_type} patterns", self.Fore.GREEN)

# Integrate into main trading class
class FullyAutonomousAITrader(SelfLearningAITrader):
    def __init__(self):
        super().__init__()
        # Your existing initialization code
        self.ai_opened_trades = {}
        self.real_trade_history = []
        
    def add_trade_to_history(self, trade_data):
        """Override to include learning from every trade"""
        super().add_trade_to_history(trade_data)  # Your existing code
        
        # Learn from this trade (especially if it's a loss)
        self.learn_from_mistake(trade_data)
        
        # Update performance stats
        self.performance_stats['total_trades'] += 1
        if trade_data.get('pnl', 0) > 0:
            self.performance_stats['winning_trades'] += 1

    def run_trading_cycle(self):
        """Override trading cycle to use learning-enhanced decisions"""
        try:
            # Monitor positions
            self.monitor_positions()
            self.display_dashboard()
            
            # Show learning progress every 10 cycles
            if hasattr(self, 'cycle_count') and self.cycle_count % 10 == 0:
                self.show_learning_progress()
            
            self.print_color(f"\n🔍 SELF-LEARNING AI SCANNING MARKETS...", self.Fore.BLUE + self.Style.BRIGHT)
            
            qualified_signals = 0
            for pair in self.available_pairs:
                if self.available_budget > 100:
                    market_data = self.get_price_history(pair)
                    
                    # Use learning-enhanced AI decision
                    ai_decision = self.get_ai_decision_with_learning(pair, market_data)
                    
                    if ai_decision["decision"] != "HOLD" and ai_decision["position_size_usd"] > 0:
                        qualified_signals += 1
                        
                        # Execute trade
                        success = self.execute_ai_trade(pair, ai_decision)
                        if success:
                            time.sleep(2)
                
            if qualified_signals == 0:
                self.print_color("No qualified signals after mistake filtering", self.Fore.YELLOW)
                
        except Exception as e:
            self.print_color(f"Trading cycle error: {e}", self.Fore.RED)
