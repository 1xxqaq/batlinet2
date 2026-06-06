import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from pathlib import Path

from src.builders import MODELS
from src.data.databundle import DataBundle, Dataset
from src.models.rul_predictors.cnn import ConvModule

from ..nn_model import NNModel


class DiffDataset(Dataset):
    def __init__(self, cycle_diff_feature, raw_feature, label):
        self.feature = cycle_diff_feature
        self.raw_feature = raw_feature
        self.label = label

    def __getitem__(self, indx):
        return {
            'feature': self.feature[indx],
            'label': self.label[indx],
            'raw_feature': self.raw_feature[indx]
        }


@torch.no_grad()
def smoothing(feature):
    med = feature.median(-1)[0].unsqueeze(-1).expand(*feature.shape)
    med_diff = (feature - med).abs()
    med_diff_std = med_diff.std(-1, keepdim=True).expand(*feature.shape)
    mask = med_diff > med_diff_std * 3
    feature[mask] = 0.
    return feature


@MODELS.register()
class BatLiNetRULPredictor(NNModel):
    def __init__(self,
                 in_channels: int,
                 channels: int,
                 input_height: int,
                 input_width: int,
                 alpha: float = 0.5,
                 kernel_size: int = 3,
                 diff_base: int = 10,
                 train_support_size: int = None,
                 test_support_size: int = None,
                 gradient_accumulation_steps: int = 1,
                 support_size: int = 1,
                 lr: float = 1e-3,
                 act_fn: str = 'relu',
                 support_aggregation: str = 'original',
                 score_head_type: str = 'linear',
                 score_hidden_channels: int = None,
                 score_dropout: float = 0.0,
                 score_input_mode: str = 'relation',
                 score_temperature: float = 1.0,
                 teacher_temperature: float = 1.0,
                 score_loss_weight: float = 0.0,
                 ranking_loss_weight: float = 0.0,
                 ranking_error_margin: float = 0.0,
                 residual_loss_weight: float = 0.0,
                 residual_filter_keep_ratio: float = 0.5,
                 residual_filter_detach_input: bool = True,
                 context_hidden_channels: int = None,
                 context_num_layers: int = 1,
                 context_num_heads: int = 4,
                 context_dropout: float = 0.0,
                 warmup_epochs: int = 0,
                 fixed_test_support_index_path: str = None,
                 filter_cycles: bool = True,
                 features_to_drop: list = None,
                 cycles_to_drop: list = None,
                 return_pointwise_predictions: bool = False,
                 seed: int = 0,
                 **kwargs):
        NNModel.__init__(self, **kwargs)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if input_height < kernel_size[0]:
            kernel_size = (input_height, kernel_size[1])
        if input_width < kernel_size[1]:
            kernel_size = (kernel_size[0], input_width)

        self.alpha = alpha
        self.channels = channels
        self.diff_base = diff_base
        self.train_support_size = train_support_size or support_size
        self.test_support_size = test_support_size or support_size
        self.grad_accum_steps = gradient_accumulation_steps
        self.support_aggregation = support_aggregation
        if score_temperature <= 0:
            raise ValueError('score_temperature must be positive.')
        if teacher_temperature <= 0:
            raise ValueError('teacher_temperature must be positive.')
        if score_loss_weight < 0:
            raise ValueError('score_loss_weight must be non-negative.')
        if ranking_loss_weight < 0:
            raise ValueError('ranking_loss_weight must be non-negative.')
        if ranking_error_margin < 0:
            raise ValueError('ranking_error_margin must be non-negative.')
        if residual_loss_weight < 0:
            raise ValueError('residual_loss_weight must be non-negative.')
        if residual_filter_keep_ratio <= 0 or residual_filter_keep_ratio > 1:
            raise ValueError('residual_filter_keep_ratio must be in (0, 1].')
        if context_num_layers <= 0:
            raise ValueError('context_num_layers must be positive.')
        if context_num_heads <= 0:
            raise ValueError('context_num_heads must be positive.')
        if context_dropout < 0 or context_dropout >= 1:
            raise ValueError('context_dropout must be in [0, 1).')
        if warmup_epochs < 0:
            raise ValueError('warmup_epochs must be non-negative.')
        if score_dropout < 0 or score_dropout >= 1:
            raise ValueError('score_dropout must be in [0, 1).')
        if score_input_mode not in (
            'relation',
            'relation_prediction_label',
            'relation_support_prediction_label',
            'context_relation_support_prediction_label',
        ):
            raise ValueError(f'Unknown score_input_mode: {score_input_mode}')
        self.score_input_mode = score_input_mode
        self.score_temperature = score_temperature
        self.teacher_temperature = teacher_temperature
        self.score_loss_weight = score_loss_weight
        self.ranking_loss_weight = ranking_loss_weight
        self.ranking_error_margin = ranking_error_margin
        self.residual_loss_weight = residual_loss_weight
        self.residual_filter_keep_ratio = residual_filter_keep_ratio
        self.residual_filter_detach_input = residual_filter_detach_input
        self.warmup_epochs = warmup_epochs
        self.fixed_test_support_index_path = fixed_test_support_index_path
        self._fixed_test_support_index = None
        self._current_epoch = None
        self.filter_cycles = filter_cycles
        if isinstance(features_to_drop, int):
            features_to_drop = [features_to_drop]
        self.features_to_drop = features_to_drop
        if isinstance(cycles_to_drop, int):
            cycles_to_drop = [cycles_to_drop]
        self.cycles_to_drop = cycles_to_drop
        self.return_pointwise_predictions = return_pointwise_predictions

        self.ori_module = build_module(
            in_channels, channels,
            input_height, input_width,
            kernel_size, act_fn)
        self.sup_module = build_module(
            in_channels, channels,
            input_height, input_width,
            kernel_size, act_fn)
        # Shared regressor without bias
        self.fc = nn.Linear(channels, 1, bias=False)
        if self.support_aggregation in (
            'learned_weighted',
            'supervised_weighted',
            'supervised_weighted_ranked',
            'supervised_weighted_fusion_ranked',
            'supervised_weighted_context_ranked',
            'rrn_residual_filter',
        ):
            score_input_dim = self.get_score_input_dim()
            self.score_head = build_score_head(
                score_input_dim, score_head_type, score_hidden_channels,
                score_dropout, context_hidden_channels, context_num_layers,
                context_num_heads, context_dropout)
        elif self.support_aggregation not in ('original', 'mean', 'median'):
            raise ValueError(
                f'Unknown support_aggregation: {self.support_aggregation}')
        self.lr = lr
        self.seed = seed

    def forward(self,
               feature: torch.Tensor,
               label: torch.Tensor,
               support_feature: torch.Tensor,
               support_label: torch.Tensor,
               return_loss: bool = False,
               epoch: int = None):
        y_ori, y_sup, y_sup_agg, weight, score = self.compute_prediction_components(
            feature, support_feature, support_label, epoch=epoch)

        if self.return_pointwise_predictions:
            return y_ori, y_sup

        if return_loss:
            loss = sum([
                (1. - self.alpha) * mse(y_ori, label),
                self.alpha * mse(y_sup_agg, label)
            ])
            if (
                self.support_aggregation == 'rrn_residual_filter'
                and score is not None
                and self.residual_loss_weight > 0
            ):
                loss = loss + self.residual_loss_weight * \
                    self.residual_prediction_loss(y_ori, y_sup, label, score)
            if (
                self.support_aggregation in (
                    'supervised_weighted',
                    'supervised_weighted_ranked',
                    'supervised_weighted_fusion_ranked',
                    'supervised_weighted_context_ranked',
                )
                and weight is not None
                and self.score_loss_weight > 0
            ):
                loss = loss + self.score_loss_weight * \
                    self.score_supervision_loss(y_ori, y_sup, label, weight)
            if (
                self.support_aggregation in (
                    'supervised_weighted_ranked',
                    'supervised_weighted_fusion_ranked',
                    'supervised_weighted_context_ranked',
                )
                and score is not None
                and self.ranking_loss_weight > 0
            ):
                loss = loss + self.ranking_loss_weight * \
                    self.score_ranking_loss(y_ori, y_sup, label, score)
            return loss

        return (1. - self.alpha) * y_ori + self.alpha * y_sup_agg

    def compute_prediction_components(self,
                                      feature: torch.Tensor,
                                      support_feature: torch.Tensor,
                                      support_label: torch.Tensor,
                                      epoch: int = None):
        B, S, C, H, W = support_feature.size()

        x_ori = self.ori_module(feature)
        x_sup = self.sup_module(support_feature.view(-1, C, H, W))
        x_sup = x_sup.view(B, S, self.channels)

        y_ori = self.fc(x_ori.view(B, self.channels)).view(-1)
        y_sup = self.fc(x_sup).view(B, S)
        y_sup += support_label.view(B, S)
        y_sup_agg, weight, score = self.aggregate_support_predictions(
            x_sup, x_ori, y_sup, y_ori, support_label, epoch=epoch)
        return y_ori, y_sup, y_sup_agg, weight, score

    def aggregate_support_predictions(self,
                                      x_sup,
                                      x_ori,
                                      y_sup,
                                      y_ori,
                                      support_label,
                                      epoch=None):
        if (
            self.support_aggregation == 'rrn_residual_filter'
        ):
            score_input = self.build_score_input(
                x_sup, x_ori, y_sup, y_ori, support_label)
            if self.residual_filter_detach_input:
                score_input = score_input.detach()
            predicted_residual = F.softplus(
                self.score_head(score_input).squeeze(-1))
            if self.use_weighted_aggregation(epoch):
                y_sup_agg, weight = self.residual_filter_aggregate(
                    y_sup, predicted_residual)
                return y_sup_agg, weight, predicted_residual
            if self.training:
                return y_sup.mean(1).view(-1), None, predicted_residual
            return y_sup.median(1)[0].view(-1), None, predicted_residual

        if (
            self.support_aggregation in (
                'learned_weighted',
                'supervised_weighted',
                'supervised_weighted_ranked',
                'supervised_weighted_fusion_ranked',
                'supervised_weighted_context_ranked',
            )
            and self.use_weighted_aggregation(epoch)
        ):
            score_input = self.build_score_input(
                x_sup, x_ori, y_sup, y_ori, support_label)
            score = self.score_head(score_input).squeeze(-1)
            weight = torch.softmax(score / self.score_temperature, dim=1)
            return (weight * y_sup).sum(1).view(-1), weight, score

        if self.support_aggregation == 'mean':
            return y_sup.mean(1).view(-1), None, None

        if self.support_aggregation == 'median':
            return y_sup.median(1)[0].view(-1), None, None

        if self.training:
            return y_sup.mean(1).view(-1), None, None

        return y_sup.median(1)[0].view(-1), None, None

    def use_weighted_aggregation(self, epoch=None):
        if self.support_aggregation not in (
            'supervised_weighted',
            'supervised_weighted_ranked',
            'supervised_weighted_fusion_ranked',
            'supervised_weighted_context_ranked',
            'rrn_residual_filter',
        ):
            return True
        current_epoch = self._current_epoch if epoch is None else epoch
        if current_epoch is None:
            return True
        return current_epoch >= self.warmup_epochs

    def get_score_input_dim(self):
        if self.score_input_mode == 'relation':
            return self.channels
        if self.score_input_mode == 'relation_support_prediction_label':
            # y_sup, support_label, y_sup-support_label, |y_sup-support_label|
            return self.channels + 4
        if self.score_input_mode == 'context_relation_support_prediction_label':
            # x_ori plus relation-support scalar features.
            return self.channels * 2 + 4
        # y_sup, y_ori, y_sup-y_ori, |y_sup-y_ori|,
        # support_label, support_label-y_ori, |support_label-y_ori|
        return self.channels + 7

    def build_score_input(self, x_sup, x_ori, y_sup, y_ori, support_label):
        if self.score_input_mode == 'relation':
            return x_sup

        B, S, _ = x_sup.size()
        y_sup_input = y_sup.detach().unsqueeze(-1)
        support_label_input = support_label.detach().view(B, S, 1)
        if self.score_input_mode == 'relation_support_prediction_label':
            support_prediction_delta = y_sup_input - support_label_input
            scalar_features = [
                y_sup_input,
                support_label_input,
                support_prediction_delta,
                support_prediction_delta.abs(),
            ]
            return torch.cat([x_sup, *scalar_features], dim=-1)

        if self.score_input_mode == 'context_relation_support_prediction_label':
            x_ori_input = x_ori.view(B, 1, self.channels).expand(-1, S, -1)
            support_prediction_delta = y_sup_input - support_label_input
            scalar_features = [
                y_sup_input,
                support_label_input,
                support_prediction_delta,
                support_prediction_delta.abs(),
            ]
            return torch.cat([x_sup, x_ori_input, *scalar_features], dim=-1)

        y_ori_input = y_ori.detach().view(B, 1, 1).expand(-1, S, -1)
        y_sup_delta = y_sup_input - y_ori_input
        support_label_delta = support_label_input - y_ori_input
        scalar_features = [
            y_sup_input,
            y_ori_input,
            y_sup_delta,
            y_sup_delta.abs(),
            support_label_input,
            support_label_delta,
            support_label_delta.abs(),
        ]
        return torch.cat([x_sup, *scalar_features], dim=-1)

    def residual_filter_aggregate(self, y_sup, predicted_residual):
        B, S = y_sup.size()
        keep_count = int(round(S * self.residual_filter_keep_ratio))
        keep_count = min(S, max(1, keep_count))
        keep_index = torch.topk(
            predicted_residual, k=keep_count, dim=1, largest=False)[1]
        selected_y_sup = torch.gather(y_sup, 1, keep_index)
        weight = torch.zeros_like(y_sup)
        weight.scatter_(1, keep_index, 1.0 / keep_count)
        return selected_y_sup.median(1)[0].view(B), weight

    def score_supervision_loss(self, y_ori, y_sup, label, weight):
        with torch.no_grad():
            reference_prediction = self.teacher_reference_prediction(
                y_ori, y_sup)
            error = (reference_prediction - label.view(-1, 1)).abs()
            teacher_weight = torch.softmax(
                -error / self.teacher_temperature, dim=1)
            teacher_weight = torch.clamp(teacher_weight, min=1e-8)

        log_teacher = torch.log(teacher_weight)
        log_weight = torch.log(torch.clamp(weight, min=1e-8))
        return (teacher_weight * (log_teacher - log_weight)).sum(1).mean()

    def score_ranking_loss(self, y_ori, y_sup, label, score):
        with torch.no_grad():
            reference_prediction = self.teacher_reference_prediction(
                y_ori, y_sup)
            error = (reference_prediction - label.view(-1, 1)).abs()
            error_i = error.unsqueeze(2)
            error_j = error.unsqueeze(1)
            better_pair = error_i + self.ranking_error_margin < error_j

        if not better_pair.any():
            return score.sum() * 0.

        score_gap = score.unsqueeze(2) - score.unsqueeze(1)
        pair_loss = F.softplus(-score_gap)
        return pair_loss[better_pair].mean()

    def residual_prediction_loss(self, y_ori, y_sup, label, predicted_residual):
        with torch.no_grad():
            reference_prediction = (1. - self.alpha) * \
                y_ori.detach().view(-1, 1) + self.alpha * y_sup.detach()
            target_residual = (
                reference_prediction - label.detach().view(-1, 1)).abs()
        return F.smooth_l1_loss(predicted_residual, target_residual)

    def teacher_reference_prediction(self, y_ori, y_sup):
        if self.support_aggregation in (
            'supervised_weighted_fusion_ranked',
            'supervised_weighted_context_ranked',
            'rrn_residual_filter',
        ):
            y_ori = y_ori.detach().view(-1, 1)
            return (1. - self.alpha) * y_ori + self.alpha * y_sup.detach()
        return y_sup.detach()

    def fit(self, dataset: DataBundle, timestamp: str):
        self.train()
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)

        # Build a cycle diff dataset
        train_dataset = self.build_cycle_diff_dataset(dataset.train_data)
        ori_loader = DataLoader(
            train_dataset, self.train_batch_size, shuffle=False)

        latest = None
        for epoch in tqdm(range(self.train_epochs), desc='Training'):
            self._current_epoch = epoch
            self.train()

            for indx, data_batch in enumerate(ori_loader):
                x, y, raw_x = data_batch.values()
                sup_x, sup_y = self.get_support_set(
                    raw_x, dataset.train_data.feature, dataset.train_data.label)
                loss = self.forward(
                    x, y, sup_x, sup_y, return_loss=True, epoch=epoch)
                loss.backward()

                if (
                    indx == len(ori_loader) - 1
                    or (indx + 1) % self.grad_accum_steps == 0
                ):
                    optimizer.step()
                    optimizer.zero_grad()

            if (
                self.workspace is not None
                and self.checkpoint_freq is not None
                and (epoch + 1) % self.checkpoint_freq == 0
            ):
                filename = self.workspace / f'{timestamp}_seed_{self.seed}_epoch_{epoch+1}.ckpt'
                self.dump_checkpoint(filename)
                latest = filename

            if (epoch + 1) % self.evaluate_freq == 0:
                del loss, sup_x, sup_y, x, y
                pred = self.predict(dataset)
                score = dataset.evaluate(pred, 'RMSE')
                message = f'[{epoch+1}/{self.train_epochs}] RMSE {score:.2f}'
                print(message, flush=True)
                del pred

        # Create symlink latest
        if latest is not None and self.workspace is not None:
            self.link_latest_checkpoint(latest)

    @torch.no_grad()
    def predict(self,
                dataset: DataBundle,
                return_diagnostics: bool = False) -> torch.Tensor:
        self.eval()
        # Build a cycle diff dataset
        test_dataset = self.build_cycle_diff_dataset(dataset.test_data)
        ori_loader = DataLoader(
            test_dataset, self.test_batch_size, shuffle=False)
        fixed_indices = self.load_fixed_test_support_indices(dataset)
        predictions = []
        diagnostics = {
            'y_ori': [],
            'y_sup': [],
            'y_sup_agg': [],
            'support_index': [],
            'support_weight': [],
            'support_score': [],
        } if return_diagnostics else None
        offset = 0
        for indx, data_batch in enumerate(ori_loader):
            x, y, raw_x = data_batch.values()
            batch_fixed_indices = None
            if fixed_indices is not None:
                batch_fixed_indices = fixed_indices[offset:offset + len(x)]
                offset += len(x)
            sup_x, sup_y, sup_indx = self.get_support_set(
                raw_x,
                dataset.train_data.feature,
                dataset.train_data.label,
                fixed_indices=batch_fixed_indices,
                return_indices=True)
            if return_diagnostics:
                y_ori, y_sup, y_sup_agg, weight, score = self.compute_prediction_components(
                    x, sup_x, sup_y)
                pred = (1. - self.alpha) * y_ori + self.alpha * y_sup_agg
                predictions.append(pred)
                diagnostics['y_ori'].append(y_ori)
                diagnostics['y_sup'].append(y_sup)
                diagnostics['y_sup_agg'].append(y_sup_agg)
                diagnostics['support_index'].append(sup_indx)
                if weight is not None:
                    diagnostics['support_weight'].append(weight)
                if score is not None:
                    diagnostics['support_score'].append(score)
            else:
                predictions.append(self.forward(x, y, sup_x, sup_y))
        if self.return_pointwise_predictions:
            predictions = (
                torch.cat([x[0] for x in predictions]),
                torch.cat([x[1] for x in predictions]),
            )
        else:
            predictions = torch.cat(predictions)
        if not return_diagnostics:
            return predictions

        support_weight = None
        if diagnostics['support_weight']:
            support_weight = torch.cat(diagnostics['support_weight'])
        support_score = None
        if diagnostics['support_score']:
            support_score = torch.cat(diagnostics['support_score'])

        diagnostics = {
            'y_ori': torch.cat(diagnostics['y_ori']),
            'y_sup': torch.cat(diagnostics['y_sup']),
            'y_sup_agg': torch.cat(diagnostics['y_sup_agg']),
            'support_index': torch.cat(diagnostics['support_index']),
            'support_weight': support_weight,
            'support_score': support_score,
        }
        return predictions, diagnostics

    @torch.no_grad()
    def build_cycle_diff_dataset(self, dataset: Dataset):
        feature = dataset.feature - dataset.feature[:, :, [self.diff_base]]
        raw_feature = dataset.feature
        if self.features_to_drop is not None:
            mask = [x for x in range(feature.size(1))
                    if x not in self.features_to_drop]
            feature = feature[:, mask].contiguous()
            raw_feature = raw_feature[:, mask].contiguous()
        if self.cycles_to_drop is not None:
            feature[:, :, self.cycles_to_drop] = 0.
            raw_feature[:, :, self.cycles_to_drop] = 0.
        feature = self._clean_feature(feature)
        raw_feature = self._filter_cycles(raw_feature)
        return DiffDataset(feature, raw_feature, dataset.label)

    @torch.no_grad()
    def get_support_set(self,
                        x,
                        sup_feat,
                        sup_label,
                        fixed_indices=None,
                        return_indices: bool = False):
        if self.features_to_drop is not None:
            mask = [i for i in range(sup_feat.size(1))
                    if i not in self.features_to_drop]
            sup_feat = sup_feat[:, mask].contiguous()
        if self.cycles_to_drop is not None:
            sup_feat[:, :, :, self.cycles_to_drop] = 0.
        if fixed_indices is not None:
            indx = fixed_indices.to(x.device).long().contiguous()
            if indx.dim() != 2 or indx.size(0) != len(x):
                raise ValueError('fixed_indices must have shape [batch, support_size].')
        else:
            if self.training:
                size = (len(x) * self.train_support_size,)
            else:
                size = (len(x) * self.test_support_size,)
            indx = torch.randint(len(sup_feat), size, device=x.device)
        B, C, H, W = x.size()
        flat_indx = indx.view(-1)
        feature = x.unsqueeze(1) - sup_feat[flat_indx].view(B, -1, C, H, W)
        label = sup_label[flat_indx].view(B, -1)
        feature = self._clean_feature(feature)
        if return_indices:
            return feature, label, indx.view(B, -1)
        return feature, label

    def load_fixed_test_support_indices(self, dataset: DataBundle):
        if self.fixed_test_support_index_path is None:
            return None
        if self._fixed_test_support_index is None:
            path = Path(self.fixed_test_support_index_path)
            payload = torch.load(path, map_location='cpu')
            if isinstance(payload, dict):
                indices = payload.get('indices')
            else:
                indices = payload
            if indices is None:
                raise ValueError(
                    f'No support indices found in {self.fixed_test_support_index_path}.')
            if indices.dim() != 2:
                raise ValueError('Fixed test support indices must be a 2D tensor.')
            if indices.size(0) != len(dataset.test_data):
                raise ValueError(
                    'Fixed test support protocol does not match the number of test samples.')
            if indices.size(1) != self.test_support_size:
                raise ValueError(
                    'Fixed test support protocol does not match test_support_size.')
            self._fixed_test_support_index = indices.long().contiguous()
        return self._fixed_test_support_index

    def _clean_feature(self, feature):
        num = 50
        feature[..., :num] = smoothing(feature[..., :num])
        feature[..., -num:] = smoothing(feature[..., -num:])
        feature = remove_glitches(feature)
        # Filter problematic cycles using Hampel filter
        feature = self._filter_cycles(feature)
        return feature

    def _filter_cycles(self, feature):
        if not self.filter_cycles:
            return feature
        feature = feature.clone()

        # Filter the cycles with its max value too large
        max_val = feature.abs().amax(-1)
        max_val_med = max_val.median(-1, keepdim=True)[0]
        max_val_diff = (max_val - max_val_med).abs()
        mask = max_val_diff > max_val_diff.std(-1, keepdim=True) * 5

        # Filter the cycles with its mean deviating from other cycles
        mean_val = feature.mean(-1)
        mean_val_med = mean_val.median(-1, keepdim=True)[0]
        mean_val_diff = (mean_val - mean_val_med).abs()
        mask |= mean_val_diff > mean_val_diff.std(-1, keepdim=True) * 5

        # Fill with zero
        feature[mask] = 0.

        return feature


def _remove_glitches(x, width, threshold):
    left_element = torch.roll(x, shifts=1, dims=-1)
    right_element = torch.roll(x, shifts=-1, dims=-1)
    diff_with_left_element = (left_element - x).abs()
    diff_with_right_element = (right_element - x).abs()

    # diff_with_left_element[..., 0] = 0.
    # diff_with_right_element[..., -1] = 0.

    ths = diff_with_left_element.std(-1, keepdim=True) * threshold
    non_smooth_on_left = diff_with_left_element > ths
    ths = diff_with_right_element.std(-1, keepdim=True) * threshold
    non_smooth_on_right = diff_with_right_element > ths
    for _ in range(width):
        non_smooth_on_left |= torch.roll(
            non_smooth_on_left, shifts=1, dims=-1)
        non_smooth_on_right |= torch.roll(
            non_smooth_on_right, shifts=-1, dims=-1)
    to_smooth = non_smooth_on_left & non_smooth_on_right
    x[to_smooth] = 0.
    return x


def remove_glitches(data, width=25, threshold=3):
    shape = data.shape
    data = data.view(-1, *shape[-3:])
    for i in range(len(data)):
        data[i] = _remove_glitches(data[i], width, threshold)
    data = data.view(shape)
    return data


def build_score_head(input_dim,
                     score_head_type,
                     score_hidden_channels,
                     score_dropout=0.0,
                     context_hidden_channels=None,
                     context_num_layers=1,
                     context_num_heads=4,
                     context_dropout=0.0):
    if score_head_type == 'linear':
        return nn.Linear(input_dim, 1)
    if score_head_type == 'mlp':
        score_hidden_channels = score_hidden_channels or max(input_dim // 2, 1)
        return nn.Sequential(
            nn.Linear(input_dim, score_hidden_channels),
            nn.ReLU(),
            nn.Linear(score_hidden_channels, 1)
        )
    if score_head_type == 'mlp_ln_gelu':
        score_hidden_channels = score_hidden_channels or max(input_dim, 1)
        return nn.Sequential(
            nn.Linear(input_dim, score_hidden_channels),
            nn.LayerNorm(score_hidden_channels),
            nn.GELU(),
            nn.Dropout(score_dropout),
            nn.Linear(score_hidden_channels, score_hidden_channels),
            nn.GELU(),
            nn.Dropout(score_dropout),
            nn.Linear(score_hidden_channels, 1)
        )
    if score_head_type == 'transformer_set':
        hidden_channels = context_hidden_channels or score_hidden_channels
        hidden_channels = hidden_channels or max(input_dim, 1)
        return SetContextScoreHead(
            input_dim=input_dim,
            hidden_channels=hidden_channels,
            num_layers=context_num_layers,
            num_heads=context_num_heads,
            dropout=context_dropout)
    raise ValueError(f'Unknown score_head_type: {score_head_type}')


class SetContextScoreHead(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_channels,
                 num_layers,
                 num_heads,
                 dropout):
        super().__init__()
        if hidden_channels % num_heads != 0:
            raise ValueError(
                'context hidden_channels must be divisible by num_heads.')
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=num_heads,
            dim_feedforward=hidden_channels * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True)
        self.context_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.score_proj = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, score_input):
        token = self.input_proj(score_input)
        token = self.context_encoder(token)
        return self.score_proj(token)


def build_module(
    in_channels, channels, input_height, input_width, kernel_size, act_fn
) -> nn.Module:
    encoder = ConvModule(in_channels, channels, kernel_size, act_fn)
    H, W = encoder.output_shape(input_height, input_width)
    proj = nn.Conv2d(channels, channels, (H, W))
    return nn.Sequential(encoder, proj, nn.ReLU())


def mse(pred, label):
    return ((pred.view(-1) - label.view(-1)) ** 2).mean()
