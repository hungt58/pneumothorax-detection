from build_dataset import BalancedBinaryBatchSampler


def test_balanced_sampler_returns_half_positive_half_negative():
    records = [
        {"HasDisease": 1}, {"HasDisease": 1},
        {"HasDisease": 0}, {"HasDisease": 0}, {"HasDisease": 0},
    ]
    sampler = BalancedBinaryBatchSampler(records, batch_size=4, seed=42, batches_per_epoch=10)
    for batch in sampler:
        labels = [records[i]["HasDisease"] for i in batch]
        assert sum(labels) == 2
        assert len(labels) == 4
