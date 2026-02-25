import train


def test_build_datasets_uses_train_label_space() -> None:
    docs = [
        {"text": "train one", "topics": ["earn", "acq"], "lewissplit": "TRAIN", "topics_attr": "YES"},
        {"text": "train two", "topics": ["earn"], "lewissplit": "TRAIN", "topics_attr": "YES"},
        {"text": "test one", "topics": ["earn", "grain"], "lewissplit": "TEST", "topics_attr": "YES"},
        {"text": "skip", "topics": ["earn"], "lewissplit": "TEST", "topics_attr": "NO"},
    ]

    train_ds, test_ds, labels = train.build_datasets(docs)

    assert labels == ["acq", "earn"]
    assert len(train_ds) == 2
    assert len(test_ds) == 1
    # "grain" is not in training label space and should be ignored.
    assert test_ds[0]["labels"] == [0.0, 1.0]

