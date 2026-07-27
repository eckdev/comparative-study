from .metrics import bootstrap_ci, group_rows, landmark_rows, summarize_errors, write_outliers, write_predictions, write_rows


def save_fold_outputs(output_dir, samples, sample_indices, pred, expert, errors, confidence):
    write_predictions(output_dir / "predictions_test.csv", samples, sample_indices, pred, expert, errors, confidence)
    write_rows(output_dir / "landmark_metrics.csv", landmark_rows(errors))
    write_rows(output_dir / "group_metrics.csv", group_rows(samples, sample_indices, errors))
    write_outliers(output_dir / "outlier_samples.csv", samples, sample_indices, errors)
    return {
        "overall": summarize_errors(errors),
        "core20": summarize_errors(errors[:, [i for i in range(23) if i not in {0, 21, 22}]]),
        "hard_landmarks": summarize_errors(errors[:, [0, 21, 22]]),
        "bootstrap_ale": bootstrap_ci(errors),
    }
