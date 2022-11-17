# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import tensorflow as tf

import xpu
from data_utils.input_spec import create_inputs_from_features
from model.utils import (create_model, get_loss_functions, get_metrics, get_tf_dataset)
from utils import (convert_loss_and_metric_reductions_to_fp32, get_optimizer, str_dtype_to_tf_dtype)


def predict(preprocessed_dataset, checkpoint_path, fold, cfg):
    """
    Function to run inference and give predictions of molecule samples.
    Args:
        preprocessed_dataset: dataset object, preprocessed dataset with all the features needed for inference.
        checkpoint_path: string, path to the checkpoint.
        fold: string, name of the fold, choose from ["valid", "test-dev" and "test-challenge"].
        cfg: config object, configurations of the model and inference.
    """
    tf.keras.mixed_precision.set_global_policy(cfg.model.dtype)

    print("Creating input specification...")
    input_spec = create_inputs_from_features(dataset=preprocessed_dataset, cfg=cfg, fold="test-dev")

    losses, loss_weights = get_loss_functions(preprocessed_dataset, cfg)
    metrics = get_metrics(preprocessed_dataset.denormalize, cfg)
    optimizer_options = dict(
        name=cfg.model.opt.lower(),
        learning_rate=cfg.model.lr,
        l2_regularization=cfg.model.l2_regularization,
        dtype=str_dtype_to_tf_dtype(cfg.model.dtype),
        m_dtype=str_dtype_to_tf_dtype(cfg.model.adam_m_dtype),
        v_dtype=str_dtype_to_tf_dtype(cfg.model.adam_v_dtype),
        clip_value=cfg.model.grad_clip_value,
        loss_scale=cfg.model.loss_scaling,
        gradient_accumulation_factor=cfg.ipu_opts.gradient_accumulation_factor,
        replicas=cfg.ipu_opts.replicas,
        outline_apply_gradients=not cfg.ipu_opts.offload_optimizer_state,  # bug where outlining causes issues
    )

    print("Configuring the IPUs...")
    strategy = xpu.configure_and_get_strategy(num_replicas=1,
                                              num_ipus_per_replica=1,
                                              stochastic_rounding=False,
                                              cfg=cfg)

    with strategy.scope():
        print("Creating TensorFlow dataset from preprocessed dataset...")
        batch_generator, ground_truth_and_masks = get_tf_dataset(preprocessed_dataset=preprocessed_dataset,
                                                                 split_name=fold,
                                                                 shuffle=False,
                                                                 options=cfg,
                                                                 input_spec=input_spec)
        ds = batch_generator.get_tf_dataset()
        ground_truth, include_mask = ground_truth_and_masks
        ground_truth = ground_truth[include_mask]

        print("Constructing the model...")
        model = create_model(batch_generator, preprocessed_dataset, cfg, input_spec=input_spec)
        model.compile(optimizer=get_optimizer(**optimizer_options),
                      loss=losses,
                      loss_weights=loss_weights,
                      weighted_metrics=metrics,
                      steps_per_execution=batch_generator.batches_per_epoch)
        if cfg.model.dtype == 'float16':
            # the loss reduction is set by backend.floatx by default
            # must be forced to reduce in float32 to avoid overflow
            convert_loss_and_metric_reductions_to_fp32(model)

        if checkpoint_path is not None:
            print(f"Loading the checkpoint from {checkpoint_path}...")
            model.load_weights(checkpoint_path).expect_partial()

        print(f"Running `model.predict` to generate predictions...")

        prediction = model.predict(ds, steps=batch_generator.batches_per_epoch)

        if isinstance(prediction, list) and len(prediction) > 1:
            prediction = prediction[0]
        prediction = prediction.squeeze()

        prediction = preprocessed_dataset.denormalize(prediction)

        if len(include_mask) > len(prediction):
            include_mask = include_mask[:len(prediction)]
            ground_truth = ground_truth[:len(prediction)]

        if len(include_mask) > 1:
            include_mask = include_mask.squeeze()

        prediction = prediction[:len(include_mask)][include_mask == 1]

    return prediction, ground_truth
