"""Implements triplet loss. Mofified from the original tf_addons file
    https://github.com/tensorflow/addons/blob/v0.15.0/tensorflow_addons/losses/triplet.py
"""
import math
import tensorflow as tf
from typing import Optional, Union, Callable

# Reemplazo de los tipos antiguos de TFA
TensorLike = Union[tf.Tensor, float, list]
FloatTensorLike = Union[tf.Tensor, float, list]

def angular_distance(feature):
    """Native implementation of angular distance."""
    cosine_similarities = tf.matmul(feature, feature, transpose_b=True)
    # Evitar inestabilidad numérica en acos (NaNs)
    cosine_similarities = tf.clip_by_value(cosine_similarities, -1.0 + 1e-16, 1.0 - 1e-16)
    return tf.math.acos(cosine_similarities) / tf.constant(math.pi, dtype=feature.dtype)


@tf.function
def pairwise_distance_64bit(feature: TensorLike, squared: bool = False):
    """Computes the pairwise distance matrix with numerical stability."""
    pairwise_distances_squared = (
        tf.math.add(
            tf.math.reduce_sum(tf.math.square(feature), axis=[1], keepdims=True),
            tf.math.reduce_sum(
                tf.math.square(tf.transpose(feature)), axis=[0], keepdims=True
            ),
        )
        - 2.0 * tf.matmul(feature, tf.transpose(feature))
    )

    pairwise_distances_squared = tf.math.maximum(pairwise_distances_squared, 0.0)
    error_mask = tf.math.less_equal(pairwise_distances_squared, 0.0)

    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = tf.math.sqrt(
            pairwise_distances_squared
            + tf.cast(error_mask, dtype=tf.dtypes.float64) * 1e-16
        )

    pairwise_distances = tf.math.multiply(
        pairwise_distances,
        tf.cast(tf.math.logical_not(error_mask), dtype=tf.dtypes.float64),
    )

    num_data = tf.shape(feature)[0]
    mask_offdiagonals = tf.cast(tf.ones_like(pairwise_distances), tf.dtypes.float64) - \
      tf.cast(tf.linalg.diag(tf.ones([num_data])), tf.dtypes.float64)
    pairwise_distances = tf.math.multiply(pairwise_distances, mask_offdiagonals)
    return pairwise_distances


def _masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements."""
    axis_minimums = tf.math.reduce_min(data, dim, keepdims=True)
    masked_maximums = (
        tf.math.reduce_max(
            tf.math.multiply(data - axis_minimums, mask), dim, keepdims=True
        )
        + axis_minimums
    )
    return masked_maximums


def _masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements."""
    axis_maximums = tf.math.reduce_max(data, dim, keepdims=True)
    masked_minimums = (
        tf.math.reduce_min(
            tf.math.multiply(data - axis_maximums, mask), dim, keepdims=True
        )
        + axis_maximums
    )
    return masked_minimums


@tf.function
def triplet_semihard_loss_64bit(
    y_true: TensorLike,
    y_pred: TensorLike,
    margin: FloatTensorLike = 1.0,
    distance_metric: Union[str, Callable] = "L2",
) -> tf.Tensor:
    r"""Computes the triplet loss with semi-hard negative mining."""
    labels = tf.convert_to_tensor(y_true, name="labels")
    embeddings = tf.convert_to_tensor(y_pred, name="embeddings")

    convert_to_float64 = (
        embeddings.dtype == tf.dtypes.float16 or embeddings.dtype == tf.dtypes.bfloat16 or
        embeddings.dtype == tf.dtypes.float64
    )
    precise_embeddings = (
        tf.cast(embeddings, tf.dtypes.float64) if convert_to_float64 else embeddings
    )

    lshape = tf.shape(labels)
    labels = tf.reshape(labels, [lshape[0], 1])

    if distance_metric == "L2":
        pdist_matrix = pairwise_distance_64bit(precise_embeddings, squared=False)
    elif distance_metric == "squared-L2":
        pdist_matrix = pairwise_distance_64bit(precise_embeddings, squared=True)
    elif distance_metric == "angular":
        pdist_matrix = angular_distance(precise_embeddings)
    else:
        pdist_matrix = distance_metric(precise_embeddings)

    adjacency = tf.math.equal(labels, tf.transpose(labels))
    adjacency_not = tf.math.logical_not(adjacency)

    batch_size = tf.size(labels)

    pdist_matrix_tile = tf.tile(pdist_matrix, [batch_size, 1])
    pdist_matrix_tile = tf.cast(pdist_matrix_tile, tf.dtypes.float64)
    mask = tf.math.logical_and(
        tf.tile(adjacency_not, [batch_size, 1]),
        tf.math.greater(
            pdist_matrix_tile, tf.reshape(tf.transpose(pdist_matrix), [-1, 1])
        ),
    )
    mask_final = tf.reshape(
        tf.math.greater(
            tf.math.reduce_sum(
                tf.cast(mask, dtype=tf.dtypes.float64), 1, keepdims=True
            ),
            0.0,
        ),
        [batch_size, batch_size],
    )
    mask_final = tf.transpose(mask_final)

    adjacency_not = tf.cast(adjacency_not, dtype=tf.dtypes.float64)
    mask = tf.cast(mask, dtype=tf.dtypes.float64)

    negatives_outside = tf.reshape(
        _masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size]
    )
    negatives_outside = tf.transpose(negatives_outside)

    negatives_inside = tf.tile(
        _masked_maximum(pdist_matrix, adjacency_not), [1, batch_size]
    )
    semi_hard_negatives = tf.where(mask_final, negatives_outside, negatives_inside)

    loss_mat = tf.math.add(tf.cast(margin, tf.dtypes.float64), pdist_matrix - semi_hard_negatives)

    mask_positives = tf.cast(adjacency, dtype=tf.dtypes.float64) - tf.cast(tf.linalg.diag(
        tf.ones([batch_size])), dtype=tf.dtypes.float64)

    num_positives = tf.math.reduce_sum(mask_positives)

    triplet_loss = tf.math.truediv(
        tf.math.reduce_sum(
            tf.math.maximum(tf.math.multiply(loss_mat, mask_positives), 0.0)
        ),
        num_positives,
    )

    if convert_to_float64:
        return tf.cast(triplet_loss, embeddings.dtype)
    else:
        return triplet_loss


@tf.function
def triplet_hard_loss_64bit(
    y_true: TensorLike,
    y_pred: TensorLike,
    margin: FloatTensorLike = 1.0,
    soft: bool = False,
    distance_metric: Union[str, Callable] = "L2",
) -> tf.Tensor:
    r"""Computes the triplet loss with hard negative and hard positive mining."""
    labels = tf.convert_to_tensor(y_true, name="labels")
    embeddings = tf.convert_to_tensor(y_pred, name="embeddings")

    convert_to_float64 = (
        embeddings.dtype == tf.dtypes.float16 or embeddings.dtype == tf.dtypes.bfloat16 or
        embeddings.dtype == tf.dtypes.float64
    )
    precise_embeddings = (
        tf.cast(embeddings, tf.dtypes.float64) if convert_to_float64 else embeddings
    )

    lshape = tf.shape(labels)
    labels = tf.reshape(labels, [lshape[0], 1])

    if distance_metric == "L2":
        pdist_matrix = pairwise_distance_64bit(precise_embeddings, squared=False)
    elif distance_metric == "squared-L2":
        pdist_matrix = pairwise_distance_64bit(precise_embeddings, squared=True)
    elif distance_metric == "angular":
        pdist_matrix = angular_distance(precise_embeddings)
    else:
        pdist_matrix = distance_metric(precise_embeddings)

    adjacency = tf.math.equal(labels, tf.transpose(labels))
    adjacency_not = tf.math.logical_not(adjacency)

    adjacency_not = tf.cast(adjacency_not, dtype=tf.dtypes.float64)
    hard_negatives = _masked_minimum(pdist_matrix, adjacency_not)

    batch_size = tf.size(labels)

    adjacency = tf.cast(adjacency, dtype=tf.dtypes.float64)

    mask_positives = tf.cast(adjacency, dtype=tf.dtypes.float64) - tf.cast(tf.linalg.diag(
        tf.ones([batch_size])), dtype=tf.dtypes.float64)

    hard_positives = _masked_maximum(pdist_matrix, mask_positives)

    if soft:
        triplet_loss = tf.math.log1p(tf.math.exp(hard_positives - hard_negatives))
    else:
        triplet_loss = tf.maximum(hard_positives - hard_negatives + margin, 0.0)

    triplet_loss = tf.reduce_mean(triplet_loss)

    if convert_to_float64:
        return tf.cast(triplet_loss, embeddings.dtype)
    else:
        return triplet_loss


class TripletSemiHardLoss_64bit(tf.keras.losses.Loss):
    """Native Keras Loss replacement for TripletSemiHardLoss."""
    def __init__(
        self,
        margin: FloatTensorLike = 1.0,
        distance_metric: Union[str, Callable] = "L2",
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE, **kwargs)
        self.margin = margin
        self.distance_metric = distance_metric

    def call(self, y_true, y_pred):
        return triplet_semihard_loss_64bit(
            y_true, y_pred, margin=self.margin, distance_metric=self.distance_metric
        )


class TripletHardLoss64bit(tf.keras.losses.Loss):
    """Native Keras Loss replacement for TripletHardLoss."""
    def __init__(
        self,
        margin: FloatTensorLike = 1.0,
        soft: bool = False,
        distance_metric: Union[str, Callable] = "L2",
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE, **kwargs)
        self.margin = margin
        self.soft = soft
        self.distance_metric = distance_metric

    def call(self, y_true, y_pred):
        return triplet_hard_loss_64bit(
            y_true, y_pred, margin=self.margin, soft=self.soft, distance_metric=self.distance_metric
        )
