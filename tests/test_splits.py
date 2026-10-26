import pandas as pd

from ml.splits import train_validation_split


def test_train_validation_split_keeps_time_blocks_disjoint():
    df = pd.DataFrame(
        {"block": [0, 0, 1, 1, 2, 2, 3, 3]},
        index=pd.Index(range(8)),
    )

    train, validation = train_validation_split(
        df, df.index, seed=7, validation_size=0.5
    )

    assert set(df.loc[train, "block"]).isdisjoint(
        set(df.loc[validation, "block"])
    )
    assert set(train).isdisjoint(set(validation))
    assert set(train) | set(validation) == set(df.index)
