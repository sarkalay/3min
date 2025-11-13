# learn_script.py - ADVANCED LEARNING WITH SAFETY FEATURES
import os
import json
import time
import math
from datetime import datetime, timedelta

class SelfLearningAITrader:
    def __init__(self):
        # Mistake history storage
        self.mistakes_history_file = "ai_trading_mistakes.json"
        self.mistakes_history = self.load_mistakes_history()
        
        # Advanced learning patterns with confidence
        self.learned_patterns = {}
        
        # Performance tracking with win/loss analysis
        self.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'common_mistakes': {},
            'improvement_areas': [],
            'learning_effectiveness': 0.0  # Track if learning actually helps
        }
        
        # Advanced learning parameters
        self.learning_config = {
            'min_mistakes_for_blocking': 3,      # Minimum mistakes before blocking
            'confidence_threshold': 0.7,         # Confidence needed to block
            'pattern_decay_days': 30,            # Older patterns matter less
            'recent_mistake_weight': 2.0,        # Recent mistakes matter more
            'adaptive_learning_rate': 0.1,       # How quickly to adapt
            'max_blocking_percentage': 0.3       # Max % of trades to block
        }
        
        # Track learning effectiveness
        self.learning_effectiveness_history = []
    
    def print_color(self, text, color=""):
        """Fallback print method for learning system"""
        print(text)
    
    def load_mistakes_history(self):
        """Load AI's mistake history with timestamp filtering"""
        try:
            if os.path.exists(self.mistakes_history_file):
                with open(self.mistakes_history_file, 'r') as f:
                    mistakes = json.load(f)
                
                # Filter out very old mistakes (older than 90 days)
                recent_mistakes = []
                cutoff_time = time.time() - (90 * 24 * 60 * 60)  # 90 days
                
                for mistake in mistakes:
                    if mistake.get('timestamp', 0) > cutoff_time:
                        recent_mistakes.append(mistake)
                
                return recent_mistakes
            return []
        except Exception as e:
            print(f"Error loading mistakes history: {e}")
            return []
    
    def calculate_pattern_confidence(self, mistake_type):
        """Calculate confidence for a mistake pattern"""
        try:
            if mistake_type not in self.learned_patterns:
                return 0.0
            
            pattern_data = self.learned_patterns[mistake_type]
            occurrences = pattern_data['occurrences']
            total_trades = self.performance_stats['total_trades']
            
            if total_trades == 0:
                return 0.0
            
            # Base confidence based on frequency
            base_confidence = min(occurrences / self.learning_config['min_mistakes_for_blocking'], 1.0)
            
            # Apply time decay - recent mistakes matter more
            if pattern_data['last_occurrence']:
                days_ago = (time.time() - pattern_data['last_occurrence']) / (24 * 60 * 60)
                time_decay = max(0, 1 - (days_ago / self.learning_config['pattern_decay_days']))
                base_confidence *= time_decay
            
            # Consider learning effectiveness
            effectiveness_bonus = self.performance_stats.get('learning_effectiveness', 0.5)
            final_confidence = base_confidence * (0.5 + effectiveness_bonus * 0.5)
            
            return min(final_confidence, 1.0)
            
        except Exception as e:
            print(f"Calculate pattern confidence error: {e}")
            return 0.0
    
    def should_avoid_trade(self, ai_decision, market_data):
        """ADVANCED: Check if trade should be blocked with confidence threshold"""
        try:
            if not self.mistakes_history:
                return False
            
            # Don't block too many trades
            total_blocked = len([m for m in self.mistakes_history if m.get('was_blocked', False)])
            blocking_rate = total_blocked / max(1, self.performance_stats['total_trades'])
            
            if blocking_rate > self.learning_config['max_blocking_percentage']:
                return False
            
            decision = ai_decision["decision"]
            reasoning = ai_decision["reasoning"].lower()
            leverage = ai_decision.get("leverage", 10)
            
            # Check each pattern with confidence threshold
            for mistake_type, pattern_data in self.learned_patterns.items():
                confidence = self.calculate_pattern_confidence(mistake_type)
                
                # Only block if confidence is high enough
                if confidence >= self.learning_config['confidence_threshold']:
                    if self.matches_mistake_pattern(ai_decision, market_data, mistake_type):
                        self.print_color(f"🚫 LEARNING: Blocking '{mistake_type}' (Confidence: {confidence:.1%})", "RED")
                        
                        # Track this blocking decision
                        self.track_blocking_decision(mistake_type, ai_decision, market_data)
                        return True
            
            return False
            
        except Exception as e:
            print(f"Should avoid trade error: {e}")
            return False
    
    def track_blocking_decision(self, mistake_type, ai_decision, market_data):
        """Track blocking decisions to measure effectiveness"""
        try:
            blocking_record = {
                'timestamp': time.time(),
                'mistake_type': mistake_type,
                'ai_decision': ai_decision,
                'market_data': {
                    'pair': market_data.get('pair'),
                    'price': market_data.get('current_price'),
                    'price_change': market_data.get('price_change', 0)
                },
                'was_blocked': True
            }
            
            self.mistakes_history.append(blocking_record)
            self.save_mistakes_history()
            
        except Exception as e:
            print(f"Track blocking decision error: {e}")
    
    def measure_learning_effectiveness(self):
        """Measure if learning is actually improving performance"""
        try:
            if len(self.mistakes_history) < 10:
                return 0.5  # Neutral effectiveness with little data
            
            # Analyze recent performance vs older performance
            recent_cutoff = time.time() - (30 * 24 * 60 * 60)  # 30 days
            old_cutoff = time.time() - (60 * 24 * 60 * 60)     # 60 days
            
            recent_mistakes = [m for m in self.mistakes_history if m.get('timestamp', 0) > recent_cutoff]
            old_mistakes = [m for m in self.mistakes_history if m.get('timestamp', 0) > old_cutoff and m.get('timestamp', 0) <= recent_cutoff]
            
            # Calculate mistake rates
            recent_rate = len(recent_mistakes) / max(1, self.performance_stats.get('recent_trades', 1))
            old_rate = len(old_mistakes) / max(1, self.performance_stats.get('old_trades', 1))
            
            if old_rate == 0:
                return 0.5
            
            # Effectiveness: lower mistake rate = better learning
            effectiveness = max(0, 1 - (recent_rate / old_rate))
            
            # Update performance stats
            self.performance_stats['learning_effectiveness'] = effectiveness
            
            return effectiveness
            
        except Exception as e:
            print(f"Measure learning effectiveness error: {e}")
            return 0.5
    
    def adaptive_learning_adjustment(self):
        """Dynamically adjust learning parameters based on effectiveness"""
        try:
            effectiveness = self.performance_stats.get('learning_effectiveness', 0.5)
            
            # Adjust confidence threshold based on effectiveness
            if effectiveness > 0.7:  # Learning is working well
                # Be more aggressive with blocking
                self.learning_config['confidence_threshold'] = max(0.5, self.learning_config['confidence_threshold'] - 0.05)
            elif effectiveness < 0.3:  # Learning is not working well
                # Be more conservative with blocking
                self.learning_config['confidence_threshold'] = min(0.9, self.learning_config['confidence_threshold'] + 0.05)
            
            # Adjust based on total experience
            total_trades = self.performance_stats['total_trades']
            if total_trades > 100:
                # With more experience, we can be more confident
                self.learning_config['min_mistakes_for_blocking'] = max(2, 3 - (total_trades // 100))
            
            self.print_color(f"🔄 LEARNING ADJUSTED: Confidence Threshold = {self.learning_config['confidence_threshold']:.2f}, Effectiveness = {effectiveness:.1%}", "MAGENTA")
            
        except Exception as e:
            print(f"Adaptive learning adjustment error: {e}")
    
    def get_learning_enhanced_prompt(self, pair, market_data):
        """SMART learning context with confidence levels"""
        try:
            if not self.mistakes_history:
                return "No learning data yet. Trading based on market analysis."
            
            learning_section = "\n🧠 SMART LEARNING INSIGHTS:\n"
            
            # Add high-confidence warnings
            high_confidence_patterns = []
            for mistake_type, pattern_data in self.learned_patterns.items():
                confidence = self.calculate_pattern_confidence(mistake_type)
                if confidence > 0.6:
                    high_confidence_patterns.append((mistake_type, confidence))
            
            if high_confidence_patterns:
                learning_section += "HIGH CONFIDENCE WARNINGS:\n"
                for pattern, confidence in sorted(high_confidence_patterns, key=lambda x: x[1], reverse=True)[:3]:
                    learning_section += f"- ⚠️  {pattern} ({(confidence*100):.0f}% confidence)\n"
            
            # Add recent lessons with context
            recent_mistakes = self.mistakes_history[-2:]
            if recent_mistakes:
                learning_section += "\nRECENT LESSONS:\n"
                for mistake in recent_mistakes:
                    lesson = mistake.get('lesson_learned', 'Unknown lesson')
                    learning_section += f"- {lesson}\n"
            
            # Add learning effectiveness
            effectiveness = self.performance_stats.get('learning_effectiveness', 0.5)
            learning_section += f"\nLEARNING EFFECTIVENESS: {(effectiveness*100):.0f}% "
            if effectiveness > 0.7:
                learning_section += "✅ (Excellent)"
            elif effectiveness > 0.5:
                learning_section += "⚠️  (Good)"
            else:
                learning_section += "❌ (Needs Improvement)"
            
            learning_section += f"\nTotal Mistakes Learned: {len(self.mistakes_history)}"
            
            return learning_section
            
        except Exception as e:
            return f"Learning analysis in progress... (Error: {e})"
    
    def learn_from_mistake(self, trade_data):
        """ADVANCED: Learn from mistakes with effectiveness tracking"""
        try:
            mistake_analysis = self.analyze_trade_mistake(trade_data)
            
            if mistake_analysis:
                # Add timestamp and effectiveness context
                mistake_analysis['timestamp'] = time.time()
                mistake_analysis['learning_effectiveness'] = self.performance_stats.get('learning_effectiveness', 0.5)
                
                # Add to mistake history
                self.mistakes_history.append(mistake_analysis)
                
                # Update performance stats
                self.performance_stats['losing_trades'] += 1
                mistake_type = mistake_analysis['mistake_type']
                self.performance_stats['common_mistakes'][mistake_type] = \
                    self.performance_stats['common_mistakes'].get(mistake_type, 0) + 1
                
                # Save and update patterns
                self.save_mistakes_history()
                self.update_learned_patterns(mistake_analysis)
                
                # Measure and adjust learning
                self.measure_learning_effectiveness()
                self.adaptive_learning_adjustment()
                
                self.print_color(f"🧠 AI LEARNING: {mistake_analysis['lesson_learned']}", "YELLOW")
                self.print_color(f"🛡️  AVOIDANCE: {mistake_analysis['avoidance_strategy']}", "CYAN")
                self.print_color(f"📊 Learning Effectiveness: {(self.performance_stats.get('learning_effectiveness', 0.5)*100):.1f}%", "MAGENTA")
                
        except Exception as e:
            print(f"Learning from mistake error: {e}")
    
    def update_learned_patterns(self, mistake_analysis):
        """Update patterns with confidence weighting"""
        try:
            mistake_type = mistake_analysis['mistake_type']
            trade_data = mistake_analysis['trade_data']
            
            if mistake_type not in self.learned_patterns:
                self.learned_patterns[mistake_type] = {
                    'occurrences': 0,
                    'total_loss': 0,
                    'avoidance_strategies': [],
                    'last_occurrence': None,
                    'pairs_affected': set(),
                    'avg_leverage': 0,
                    'first_occurrence': time.time(),
                    'confidence_history': []
                }
            
            pattern_data = self.learned_patterns[mistake_type]
            pattern_data['occurrences'] += 1
            pattern_data['total_loss'] += abs(trade_data.get('pnl', 0))
            pattern_data['last_occurrence'] = time.time()
            pattern_data['pairs_affected'].add(trade_data.get('pair', 'Unknown'))
            
            # Update average leverage
            current_leverage = trade_data.get('leverage', 1)
            if pattern_data['occurrences'] == 1:
                pattern_data['avg_leverage'] = current_leverage
            else:
                pattern_data['avg_leverage'] = (pattern_data['avg_leverage'] * (pattern_data['occurrences'] - 1) + current_leverage) / pattern_data['occurrences']
            
            # Add new avoidance strategy if different
            new_strategy = mistake_analysis['avoidance_strategy']
            if new_strategy not in pattern_data['avoidance_strategies']:
                pattern_data['avoidance_strategies'].append(new_strategy)
            
            # Update confidence history
            current_confidence = self.calculate_pattern_confidence(mistake_type)
            pattern_data['confidence_history'].append({
                'timestamp': time.time(),
                'confidence': current_confidence,
                'occurrences': pattern_data['occurrences']
            })
                
        except Exception as e:
            print(f"Update learned patterns error: {e}")
    
    # Keep all the other methods from previous version (analyze_trade_mistake, identify_mistake_type, etc.)
    # ... [Include all the previous methods here] ...
    
    def show_advanced_learning_progress(self):
        """Display advanced learning metrics"""
        try:
            total_mistakes = len(self.mistakes_history)
            common_mistakes = self.get_most_common_mistakes()
            effectiveness = self.performance_stats.get('learning_effectiveness', 0.5)
            
            self.print_color(f"\n🧠 ADVANCED LEARNING PROGRESS REPORT", "CYAN")
            self.print_color("=" * 70, "CYAN")
            self.print_color(f"Total Mistakes Learned: {total_mistakes}", "WHITE")
            self.print_color(f"Learning Effectiveness: {(effectiveness*100):.1f}%", "GREEN" if effectiveness > 0.6 else "YELLOW")
            self.print_color(f"Confidence Threshold: {self.learning_config['confidence_threshold']:.2f}", "WHITE")
            self.print_color(f"Active Patterns: {len(self.learned_patterns)}", "WHITE")
            
            self.print_color(f"\n📊 HIGH-CONFIDENCE PATTERNS:", "YELLOW")
            high_conf_patterns = []
            for pattern, data in self.learned_patterns.items():
                confidence = self.calculate_pattern_confidence(pattern)
                if confidence > 0.5:
                    high_conf_patterns.append((pattern, confidence, data['occurrences']))
            
            for pattern, confidence, occurrences in sorted(high_conf_patterns, key=lambda x: x[1], reverse=True)[:5]:
                self.print_color(f"  {pattern}: {(confidence*100):.0f}% ({occurrences} occurrences)", "WHITE")
            
            blocking_rate = len([m for m in self.mistakes_history if m.get('was_blocked', False)]) / max(1, self.performance_stats['total_trades'])
            self.print_color(f"\n🛡️  BLOCKING STATS: {(blocking_rate*100):.1f}% of trades blocked", "MAGENTA")
            
        except Exception as e:
            print(f"Show advanced learning progress error: {e}")

# Include all the previous helper methods (identify_mistake_type, extract_lesson, etc.)
# ... [Previous methods remain the same] ...
