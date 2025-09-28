import numpy as np
import time
from tqdm import tqdm

from .utils import process_attn, calc_attn_score


class AttentionDetector():
    def __init__(self, model, pos_examples=None, neg_examples=None, use_token="first_5", instruction="Say xxxxxx", threshold=0.5):
        self.name = "attention"
        self.attn_func = "normalize_sum"
        self.model = model
        self.important_heads = model.important_heads
        self.instruction = instruction
        self.use_token = use_token
        self.threshold = threshold
        
        # Tự động tìm decay rate tối ưu nếu có dữ liệu training
        if pos_examples and neg_examples and use_token == "first_5":
            self.optimal_decay = self._find_optimal_decay(pos_examples[:5], neg_examples[:5])  # Chỉ test với ít samples
        else:
            self.optimal_decay = 0.7  # Default decay rate
        if pos_examples and neg_examples:
            pos_scores, neg_scores = [], []
            for prompt in tqdm(pos_examples, desc="pos_examples"):
                _, _, attention_maps, _, input_range, generated_probs = self.model.inference(
                    self.instruction, prompt, max_output_tokens=5
                )
                pos_scores.append(self.attn2score(attention_maps, input_range))

            for prompt in tqdm(neg_examples, desc="neg_examples"):
                _, _, attention_maps, _, input_range, generated_probs = self.model.inference(
                    self.instruction, prompt, max_output_tokens=5
                )
                neg_scores.append(self.attn2score(attention_maps, input_range))

            self.threshold = (np.mean(pos_scores) + np.mean(neg_scores)) / 2

        if pos_examples and not neg_examples:
            pos_scores = []
            for prompt in tqdm(pos_examples, desc="pos_examples"):
                _, _, attention_maps, _, input_range, generated_probs = self.model.inference(
                    self.instruction, prompt, max_output_tokens=5
                )
                pos_scores.append(self.attn2score(attention_maps, input_range))

            self.threshold = np.mean(pos_scores) - 4 * np.std(pos_scores)

    def _find_optimal_decay(self, pos_examples, neg_examples):
        """Tìm decay rate tối ưu từ 0.5 đến 0.9"""
        best_decay = 0.7
        best_accuracy = 0
        
        for decay in [0.5, 0.6, 0.7, 0.8, 0.9]:
            accuracy = self._evaluate_decay(decay, pos_examples, neg_examples)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_decay = decay
        
        print(f"🎯 Optimal decay rate: {best_decay} (accuracy: {best_accuracy:.3f})")
        return best_decay
    
    def _evaluate_decay(self, decay, pos_examples, neg_examples):
        """Đánh giá accuracy với decay rate cho trước"""
        correct = 0
        total = 0
        
        for prompt in pos_examples:
            _, _, attention_maps, _, input_range, _ = self.model.inference(
                self.instruction, prompt, max_output_tokens=5
            )
            score = self._calculate_weighted_score(attention_maps, input_range, decay, verbose=False)
            if score <= 0.5:  # Temporary threshold
                correct += 1
            total += 1
        
        for prompt in neg_examples:
            _, _, attention_maps, _, input_range, _ = self.model.inference(
                self.instruction, prompt, max_output_tokens=5
            )
            score = self._calculate_weighted_score(attention_maps, input_range, decay, verbose=False)
            if score > 0.5:  # Temporary threshold
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0
    
    def _calculate_weighted_score(self, attention_maps, input_range, decay, verbose=False):
        """Tính score với exponential decay weights"""
        attention_maps = attention_maps[:5]  # Chỉ lấy 5 tokens
        scores = []
        
        for i, attention_map in enumerate(attention_maps):
            heatmap = process_attn(attention_map, input_range, self.attn_func)
            score = calc_attn_score(heatmap, self.important_heads)
            scores.append(score)
            
            if verbose:
                print(f"Token {i+1}: attention_score = {score:.4f}")
        
        if len(scores) > 0:
            # Exponential decay weights: [1.0, decay, decay^2, decay^3, decay^4]
            weights = [decay ** i for i in range(len(scores))]
            
            if verbose:
                print(f"\nWeights (decay={decay:.1f}): {[f'{w:.3f}' for w in weights]}")
                print("Weighted calculation:")
                for i, (score, weight) in enumerate(zip(scores, weights)):
                    print(f"  Token {i+1}: {score:.4f} × {weight:.3f} = {score * weight:.4f}")
            
            weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
            total_weight = sum(weights)
            final_score = weighted_sum / total_weight
            
            if verbose:
                print(f"Final score: {weighted_sum:.4f} / {total_weight:.3f} = {final_score:.4f}")
                
            return final_score
        return 0

    def attn2score(self, attention_maps, input_range):
        if self.use_token == "first":
            attention_maps = [attention_maps[0]]
            scores = []
            for attention_map in attention_maps:
                heatmap = process_attn(attention_map, input_range, self.attn_func)
                score = calc_attn_score(heatmap, self.important_heads)
                scores.append(score)
            return sum(scores) if len(scores) > 0 else 0
        
        elif self.use_token == "first_5":
            print("\n🔍 Analyzing attention for 5 tokens:")
            return self._calculate_weighted_score(attention_maps, input_range, self.optimal_decay, verbose=True)
        
        else:
            # Fallback: use all tokens with equal weights
            scores = []
            for attention_map in attention_maps:
                heatmap = process_attn(attention_map, input_range, self.attn_func)
                score = calc_attn_score(heatmap, self.important_heads)
                scores.append(score)
            return sum(scores) if len(scores) > 0 else 0

    def detect(self, data_prompt):
        # Measure generation time
        start_time = time.time()
        
        # Luôn sinh 128 tokens để có output đầy đủ, chỉ attention analysis dùng 5 token đầu
        generated_text, _, attention_maps, _, input_range, _ = self.model.inference(
            self.instruction, data_prompt, max_output_tokens=128)
        
        end_time = time.time()
        generation_time = end_time - start_time

        focus_score = self.attn2score(attention_maps, input_range)
        return bool(focus_score <= self.threshold), {
            "focus_score": focus_score,
            "generated_text": generated_text,
            "generation_time": generation_time
        }
    
    def detect_fast(self, data_prompt):
        """Fast detection using optimized inference"""
        # Measure generation time
        start_time = time.time()
        
        # Sử dụng inference_fast nếu có, nếu không fallback về inference
        # Luôn sinh 128 tokens để có output đầy đủ, chỉ attention analysis dùng 5 token đầu
        if hasattr(self.model, 'inference_fast'):
            generated_text, _, attention_maps, _, input_range, _ = self.model.inference_fast(
                self.instruction, data_prompt, max_output_tokens=128)
        else:
            generated_text, _, attention_maps, _, input_range, _ = self.model.inference(
                self.instruction, data_prompt, max_output_tokens=128)
        
        end_time = time.time()
        generation_time = end_time - start_time

        focus_score = self.attn2score(attention_maps, input_range)
        return bool(focus_score <= self.threshold), {
            "focus_score": focus_score,
            "generated_text": generated_text,
            "generation_time": generation_time
        }
