import unittest

import torch

from train_mfrag import build_property_loss


class PropertyLossTest(unittest.TestCase):
    def test_regression_is_mean_squared_error_for_small_and_large_residuals(self):
        criterion = build_property_loss('reg', 'mse')
        self.assertIsInstance(criterion, torch.nn.MSELoss)
        prediction = torch.tensor([[0.25], [2.0]], requires_grad=True)
        target = torch.zeros_like(prediction)
        loss = criterion(prediction, target)
        torch.testing.assert_close(loss, torch.tensor(2.03125))
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.tensor([[0.25], [2.0]]))

    def test_default_is_original_huber_loss(self):
        criterion = build_property_loss('reg')
        self.assertIsInstance(criterion, torch.nn.HuberLoss)
        self.assertEqual(criterion.delta, 1.0)
        prediction = torch.tensor([[0.25], [2.0]], requires_grad=True)
        loss = criterion(prediction, torch.zeros_like(prediction))
        torch.testing.assert_close(loss, torch.tensor(0.765625))
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.tensor([[0.125], [0.5]]))

    def test_huber_delta_controls_linear_branch(self):
        criterion = build_property_loss('reg', 'huber', delta=0.5)
        prediction = torch.tensor([2.0], requires_grad=True)
        loss = criterion(prediction, torch.zeros_like(prediction))
        torch.testing.assert_close(loss, torch.tensor(0.875))
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.tensor([0.5]))

    def test_unknown_regression_loss_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported regression loss'):
            build_property_loss('reg', 'unknown')

    def test_invalid_huber_delta_is_rejected(self):
        for delta in (0.0, -1.0, float('nan'), float('inf')):
            with self.subTest(delta=delta), self.assertRaisesRegex(ValueError, 'delta'):
                build_property_loss('reg', 'huber', delta)

    def test_mse_does_not_depend_on_delta(self):
        prediction = torch.tensor([0.25, 2.0])
        target = torch.zeros_like(prediction)
        for delta in (0.5, 1.0, 5.0):
            loss = build_property_loss('reg', 'mse', delta)(prediction, target)
            torch.testing.assert_close(loss, torch.tensor(2.03125))

    def test_classification_remains_binary_cross_entropy(self):
        criterion = build_property_loss('cls')
        self.assertIsInstance(criterion, torch.nn.BCEWithLogitsLoss)
        prediction = torch.tensor([-2.0, 1.0])
        target = torch.tensor([0.0, 1.0])
        torch.testing.assert_close(
            criterion(prediction, target),
            torch.nn.functional.binary_cross_entropy_with_logits(prediction, target),
        )

    def test_unknown_label_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported label mode'):
            build_property_loss('unknown')


if __name__ == '__main__':
    unittest.main()
