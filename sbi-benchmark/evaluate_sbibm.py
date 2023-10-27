
import argparse
import csv
import os.path
from os.path import join
from pathlib import Path
import math

import matplotlib.pyplot as plt
import sbibm.tasks
from sbibm.metrics import c2st, median_distance, posterior_variance_ratio, posterior_mean_error, mmd, ksd
import torch
import numpy as np
import pandas as pd
import yaml
import wandb
from torch.utils.data import Dataset
from run_sbibm import generate_dataset, load_dataset


from dingo.core.posterior_models.build_model import (
    build_model_from_kwargs,
    autocomplete_model_kwargs,
)
from dingo.core.posterior_models.base_model import test_epoch
from dingo.core.utils import build_train_and_test_loaders, RuntimeLimits


def plot_posteriors_and_log_probs(
        reference_samples,
        posterior_samples,
        reference_log_probs,
        posterior_log_probs,
        train_dir
        ):
    plt.hist(
        posterior_log_probs,
        alpha=0.2,
        label="posterior log probs",
    )
    plt.hist(
        reference_log_probs,
        alpha=0.2,
        label="reference log probs",
    )
    plt.legend()
    plt.savefig(join(train_dir, "log_probs.png"))
    plt.clf()

    plt.scatter(
        posterior_samples[:, 0],
        posterior_samples[:, 1],
        s=0.5,
        alpha=0.2,
        label="flow matching",
    )
    plt.scatter(
        reference_samples[:, 0],
        reference_samples[:, 1],
        s=0.5,
        alpha=0.2,
        label="reference",
    )
    plt.legend()
    plt.savefig(join(train_dir, "posteriors.png"))


def complete_model_evaluation(train_dir, settings, dataset, model, metrics, use_wandb=False, save_samples=True):
    task = sbibm.get_task(settings["task"]["name"])
    max_batch_size = settings["task"].get("max_batch_size", 500)
    metrics_dict = {'c2st': c2st, 'ksd': ksd, 'mmd': mmd, 'posterior_mean_error': posterior_mean_error,
                     'posterior_variance_ratio': posterior_variance_ratio, 'median_distance': median_distance}
    metrics = [m for m in metrics if m in metrics_dict.keys()]
    result_list = []

    for obs in range(1, 10):
        reference_samples = task.get_reference_posterior_samples(num_observation=obs)
        num_samples = len(reference_samples)
        reference_samples_standardized = dataset.standardize(
            reference_samples, label="theta"
        )

        observation = dataset.standardize(
            task.get_observation(num_observation=obs), label="x"
        )
        reference_log_probs = []
        for i in range(math.ceil(num_samples / max_batch_size)):
            reference_batch = reference_samples_standardized[(i*max_batch_size):((i+1)*max_batch_size)]
            # We evaluate likelihoods of the standardized data
            reference_log_probs.append(model.log_prob_batch(
                    reference_batch, observation.repeat((len(reference_batch), 1))
            ).detach())
        reference_log_probs = torch.cat(reference_log_probs, dim=0)
        # generate (num_samples * 2), to account for samples outside of the prior
        posterior_samples, posterior_log_probs = [], []

        for i in range(2 * num_samples // max_batch_size + 1):
            posterior_samples_batch, posterior_log_probs_batch = model.sample_and_log_prob_batch(
                observation.repeat((max_batch_size, 1))
            )
            posterior_samples.append(posterior_samples_batch.detach())
            posterior_log_probs.append(posterior_log_probs_batch.detach())
        posterior_samples = torch.cat(posterior_samples, dim=0)
        posterior_log_probs = torch.cat(posterior_log_probs, dim=0)

        posterior_samples = dataset.standardize(
            posterior_samples, label="theta", inverse=True
        )

        # discard samples outside the prior
        prior_mask = torch.isfinite(task.prior_dist.log_prob(posterior_samples))
        print(
            f"{(1 - torch.sum(prior_mask) / len(prior_mask)) * 100:.2f}% of the samples "
            f"lie outside of the prior. Discarding these."
        )
        posterior_samples = posterior_samples[prior_mask]
        posterior_log_probs = posterior_log_probs[prior_mask]
        n = min(len(reference_samples), len(posterior_samples))
        if len(reference_samples) > len(posterior_samples):
            print('Less posterior samples than reference samples!')
        posterior_samples = posterior_samples[:n].detach()
        posterior_log_probs = posterior_log_probs[:n].detach()
        reference_samples = reference_samples[:n].detach()
        reference_log_probs = reference_log_probs[:n].detach()

        if obs == 1:
            plot_posteriors_and_log_probs(reference_samples, posterior_samples, reference_log_probs,
                                          posterior_log_probs, train_dir)
        result = {'num_observation': obs}
        for m in metrics:
            if m == 'ksd':
                score = ksd(task, obs, posterior_samples)
            else:
                score = metrics_dict[m](posterior_samples, reference_samples).item()
            result[m] = score
        result_list.append(result)

        if save_samples:
            dir_obs = join(train_dir, str(obs).zfill(2))
            Path(dir_obs).mkdir(exist_ok=True)
            np.save(join(dir_obs, 'samples.npy'), posterior_samples)
            np.save(join(dir_obs, 'posterior_log_probs.npy'), posterior_log_probs)
            np.save(join(dir_obs, 'reference_log_probs.npy'), reference_log_probs)

    with open(
            join(train_dir, "results.csv"), "w"
    ) as f:  # You will need 'wb' mode in Python 2.x
        w = csv.DictWriter(f, result_list[0].keys())
        w.writeheader()
        w.writerows(result_list)


def compute_validation_loss(model, test_loader, train_dir):
    model.time_prior_exponent = 0
    val_loss = test_epoch(model, test_loader)
    log_probs = []
    with torch.no_grad():
        model.network.eval()
        for batch_idx, data in enumerate(test_loader):
            data = [d.to(model.device, non_blocking=True) for d in data]
            # compute loss
            log_probs_batch = model.log_prob_batch(data[0], *data[1:])
            log_probs.append(log_probs_batch)
    log_probs = torch.cat(log_probs, dim=0)
    mean_log_likelihood = torch.mean(log_probs).item()
    median_log_likelihood = torch.median(log_probs).item()
    validation_losses = [{'val_loss': val_loss, 'val_mean_log_likelihood': mean_log_likelihood,
                          'val_median_log_likelihood': median_log_likelihood}]
    df = pd.DataFrame.from_records(validation_losses)
    df.to_csv(join(train_dir, 'val_losses.csv'), index=False)


if __name__ == "__main__":
    metrics = ['c2st', 'ksd', 'mmd', 'posterior_mean_error', 'posterior_variance_ratio', 'median_distance']
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_dir", required=True, help="Base save directory for the evaluation"
    )
    parser.add_argument(
        "--dataset_dir"
    )
    parser.add_argument('--metrics', nargs='+', default=metrics)
    args = parser.parse_args()

    with open(join(args.train_dir, "settings.yaml"), "r") as f:
        settings = yaml.safe_load(f)

    use_wandb = settings["training"].get("wandb")
    if use_wandb:
        import wandb

        wandb.init(config=settings, dir=args.train_dir, **settings["training"]["wandb"])

    if args.dataset_dir is not None and os.path.exists(join(args.dataset_dir, 'x.npy')):
        dataset = load_dataset(args.dataset_dir)
    else:
        dataset = generate_dataset(
            settings,
            batch_size=settings["task"].get("batch_size", 1000),
            directory_save=args.dataset_dir
        )

    train_loader, test_loader = build_train_and_test_loaders(
        dataset,
        settings["training"]["train_fraction"],
        settings["training"]["batch_size"],
        settings["training"]["num_workers"],
    )

    model = build_model_from_kwargs(
        filename=join(args.train_dir, "best_model.pt"),
        device=settings["training"].get("device", "cpu"),
    )

    complete_model_evaluation(args.train_dir, settings, dataset, model, args.metrics, use_wandb=use_wandb)
