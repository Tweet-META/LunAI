import unittest

import torch

from rl.ppo_cnn_agent import CNNActorCritic, CNNPPOConfig, observation_scale_names


# Create a small configuration for one observation-scale baseline.
def make_config(observation_scales: str) -> CNNPPOConfig:
    return CNNPPOConfig(
        red_shape=(12, 64, 64),
        yellow_shape=(12, 16, 16),
        blue_shape=(12, 8, 8),
        player_dim=8,
        observation_scales=observation_scales,
    )


# Create one batch with every available scale.
def make_states() -> dict[str, torch.Tensor]:
    return {
        "red": torch.zeros(2, 12, 64, 64),
        "yellow": torch.zeros(2, 12, 16, 16),
        "blue": torch.zeros(2, 12, 8, 8),
        "player": torch.zeros(2, 8),
    }


class ObservationScaleBaselineTests(unittest.TestCase):
    # Check that every baseline keeps the same network capacity.
    def test_observation_scale_branches(self) -> None:
        cases = (
            ("red_only", ("red",)),
            ("red_blue", ("red", "blue")),
            ("full", ("red", "yellow", "blue")),
        )
        for mode, expected_scales in cases:
            with self.subTest(mode=mode):
                model = CNNActorCritic(make_config(mode))
                self.assertEqual(observation_scale_names(mode), expected_scales)
                for scale in ("red", "yellow", "blue"):
                    self.assertTrue(hasattr(model, f"{scale}_encoder"))

                logits, values = model(make_states())
                self.assertEqual(tuple(logits.shape), (2, 9))
                self.assertEqual(tuple(values.shape), (2,))

        parameter_counts = {
            mode: sum(parameter.numel() for parameter in CNNActorCritic(make_config(mode)).parameters())
            for mode in ("red_only", "red_blue", "full")
        }
        self.assertEqual(len(set(parameter_counts.values())), 1)

    # Check that masked scales cannot change a red-only policy output.
    def test_masked_scales_do_not_affect_output(self) -> None:
        model = CNNActorCritic(make_config("red_only"))
        model.eval()
        first_states = make_states()
        second_states = make_states()
        second_states["yellow"].fill_(1.0)
        second_states["blue"].fill_(1.0)
        with torch.no_grad():
            first_logits, first_values = model(first_states)
            second_logits, second_values = model(second_states)
        torch.testing.assert_close(first_logits, second_logits)
        torch.testing.assert_close(first_values, second_values)

    # Check that an invalid baseline name fails clearly.
    def test_invalid_observation_scales(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported observation_scales"):
            observation_scale_names("yellow_only")


if __name__ == "__main__":
    unittest.main()
