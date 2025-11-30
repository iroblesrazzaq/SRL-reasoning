"""Unit tests for SRL-Reasoning package."""

import unittest
import torch

from srl_lib.formatting import parse_model_output
from srl_lib.rewards import compute_srl_reward
from srl_lib.grpo_utils import dynamic_sampling_filter, compute_advantages


class TestParseModelOutput(unittest.TestCase):
    """Tests for parse_model_output function."""
    
    def test_valid_format_with_both_tags(self):
        """Test parsing with both <think> and </think> tags."""
        text = "<think>I need to consider the options carefully</think>The answer is B"
        thought, action = parse_model_output(text)
        self.assertEqual(thought, "I need to consider the options carefully")
        self.assertEqual(action, "The answer is B")
    
    def test_whitespace_handling(self):
        """Test that whitespace is properly stripped."""
        text = "<think>  reasoning with spaces  </think>   action with spaces   "
        thought, action = parse_model_output(text)
        self.assertEqual(thought, "reasoning with spaces")
        self.assertEqual(action, "action with spaces")
    
    def test_missing_close_tag(self):
        """Test that missing </think> returns (None, None)."""
        text = "<think>some thought without close tag"
        thought, action = parse_model_output(text)
        self.assertIsNone(thought)
        self.assertIsNone(action)
    
    def test_missing_open_tag_but_close_present(self):
        """Test edge case where <think> is missing but </think> is present."""
        text = "some content</think>final action"
        thought, action = parse_model_output(text)
        self.assertEqual(thought, "some content")
        self.assertEqual(action, "final action")
    
    def test_empty_thought(self):
        """Test with empty thought content."""
        text = "<think></think>just the action"
        thought, action = parse_model_output(text)
        self.assertEqual(thought, "")
        self.assertEqual(action, "just the action")
    
    def test_empty_action(self):
        """Test with empty action content."""
        text = "<think>only thought</think>"
        thought, action = parse_model_output(text)
        self.assertEqual(thought, "only thought")
        self.assertEqual(action, "")
    
    def test_none_input(self):
        """Test with None input."""
        thought, action = parse_model_output(None)
        self.assertIsNone(thought)
        self.assertIsNone(action)


class TestComputeSRLReward(unittest.TestCase):
    """Tests for compute_srl_reward function."""
    
    def test_exact_match_returns_one(self):
        """Test that exact match returns reward of 1.0."""
        model_completion = "<think>reasoning</think>exact answer"
        expert_target = "exact answer"
        reward = compute_srl_reward(model_completion, expert_target)
        self.assertAlmostEqual(reward, 1.0, places=5)
    
    def test_completely_different_returns_zero(self):
        """Test that completely different strings return 0.0."""
        model_completion = "<think>reasoning</think>xyz"
        expert_target = "abc"
        reward = compute_srl_reward(model_completion, expert_target)
        self.assertAlmostEqual(reward, 0.0, places=5)
    
    def test_missing_close_tag_returns_negative_one(self):
        """Test that missing </think> tag returns -1.0 penalty."""
        model_completion = "<think>reasoning without close tag"
        expert_target = "some target"
        reward = compute_srl_reward(model_completion, expert_target)
        self.assertEqual(reward, -1.0)
    
    def test_partial_match(self):
        """Test partial match returns value between 0 and 1."""
        model_completion = "<think>reasoning</think>hello world"
        expert_target = "hello there"
        reward = compute_srl_reward(model_completion, expert_target)
        self.assertGreater(reward, 0.0)
        self.assertLess(reward, 1.0)
    
    def test_empty_action_and_target(self):
        """Test that empty action and target returns 1.0."""
        model_completion = "<think>reasoning</think>"
        expert_target = ""
        reward = compute_srl_reward(model_completion, expert_target)
        self.assertEqual(reward, 1.0)
    
    def test_no_format_returns_penalty(self):
        """Test that text without </think> returns penalty."""
        model_completion = "just plain text"
        expert_target = "target"
        reward = compute_srl_reward(model_completion, expert_target)
        self.assertEqual(reward, -1.0)


class TestDynamicSamplingFilter(unittest.TestCase):
    """Tests for dynamic_sampling_filter function."""
    
    def test_identical_values_returns_false(self):
        """Test that a row with all identical values returns False in mask."""
        # Row 0: all same values (std=0)
        # Row 1: different values (std>0)
        rewards = torch.tensor([
            [0.5, 0.5, 0.5, 0.5],  # std = 0
            [0.1, 0.5, 0.8, 0.2],  # std > 0
        ])
        mask = dynamic_sampling_filter(rewards)
        self.assertFalse(mask[0].item())  # Zero variance row should be False
        self.assertTrue(mask[1].item())   # Non-zero variance row should be True
    
    def test_all_different_returns_true(self):
        """Test that rows with variance return True."""
        rewards = torch.tensor([
            [0.0, 0.5, 1.0],
            [0.2, 0.4, 0.6],
        ])
        mask = dynamic_sampling_filter(rewards)
        self.assertTrue(mask[0].item())
        self.assertTrue(mask[1].item())
    
    def test_custom_epsilon(self):
        """Test with custom epsilon threshold."""
        # Very small variance that might be below custom epsilon
        rewards = torch.tensor([
            [0.5, 0.500001, 0.5, 0.5],  # Very small std
        ])
        # With default epsilon=1e-4, this should be False
        mask = dynamic_sampling_filter(rewards, epsilon=1e-4)
        self.assertFalse(mask[0].item())
    
    def test_output_shape(self):
        """Test that output shape is correct."""
        rewards = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ])
        mask = dynamic_sampling_filter(rewards)
        self.assertEqual(mask.shape, (3,))


class TestComputeAdvantages(unittest.TestCase):
    """Tests for compute_advantages function."""
    
    def test_normalization(self):
        """Test that advantages are properly normalized."""
        rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        advantages = compute_advantages(rewards)
        
        # Mean of normalized values should be ~0
        mean = advantages.mean(dim=1)
        self.assertAlmostEqual(mean.item(), 0.0, places=5)
        
        # Std should be ~1 (accounting for sample vs population std)
        std = advantages.std(dim=1)
        self.assertAlmostEqual(std.item(), 1.0, places=1)
    
    def test_output_shape(self):
        """Test that output shape matches input shape."""
        rewards = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ])
        advantages = compute_advantages(rewards)
        self.assertEqual(advantages.shape, rewards.shape)
    
    def test_per_group_normalization(self):
        """Test that normalization is done per group (row)."""
        rewards = torch.tensor([
            [1.0, 2.0, 3.0],  # Mean = 2.0
            [10.0, 20.0, 30.0],  # Mean = 20.0
        ])
        advantages = compute_advantages(rewards)
        
        # Both rows should have mean ~0 after normalization
        row_means = advantages.mean(dim=1)
        self.assertAlmostEqual(row_means[0].item(), 0.0, places=5)
        self.assertAlmostEqual(row_means[1].item(), 0.0, places=5)
    
    def test_handles_zero_std(self):
        """Test that zero std is handled without division error."""
        rewards = torch.tensor([[1.0, 1.0, 1.0]])  # std = 0
        # Should not raise an error due to epsilon in denominator
        advantages = compute_advantages(rewards)
        self.assertEqual(advantages.shape, (1, 3))


if __name__ == "__main__":
    unittest.main()
