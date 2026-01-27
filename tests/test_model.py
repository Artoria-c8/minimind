import unittest
import torch
from model.model_minimind import MiniMindConfig, MiniMindModel

class TestMiniMindModel(unittest.TestCase):
    def test_model_initialization(self):
        # config = MiniMindConfig(
        #     hidden_size=64,
        #     num_hidden_layers=2,
        #     vocab_size=100,
        #     num_attention_heads=2,
        #     max_position_embeddings=128
        # )
        # model = MiniMindModel(config)
        # self.assertIsInstance(model, MiniMindModel)
        # self.assertEqual(model.config.hidden_size, 64)
        pass

    def test_forward_pass(self):
        config = MiniMindConfig(
            hidden_size=64,
            num_hidden_layers=2,
            vocab_size=100,
            num_attention_heads=2,
            max_position_embeddings=128
        )
        model = MiniMindModel(config)
        input_ids = torch.randint(0, 100, (1, 10))
        outputs = model(input_ids)
        # Output is (hidden_states, presents, aux_loss)
        hidden_states = outputs[0]
        self.assertEqual(hidden_states.shape, (1, 10, 64))

if __name__ == '__main__':
    unittest.main()
