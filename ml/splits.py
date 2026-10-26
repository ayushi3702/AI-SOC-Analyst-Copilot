import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit

def random_split(df, seed=42, test_size=0.2):
    """Mode A: the leaky one. Used only to demonstrate inflated scores."""
    return train_test_split(df.index, test_size=test_size, random_state=seed,
                            stratify=df["label"])

def block_split(df, seed=42, test_size=0.2):
    """Mode B: whole time blocks go to train or test, so neighbouring
    near-identical flows never straddle the boundary."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(df, groups=df["block"]))
    return df.index[tr], df.index[te]

def unseen_attack_split(df, train_days, test_days):
    """Mode C: train Mon-Wed, test Thu-Fri. Attack types in test are new."""
    return (df.index[df["day"].isin(train_days)],
            df.index[df["day"].isin(test_days)])

def get_split(mode, df, cfg):
    if mode == "random":
        return random_split(df, cfg["seed"])
    if mode == "block":
        return block_split(df, cfg["seed"])
    if mode == "unseen":
        return unseen_attack_split(df, cfg["train_days"], cfg["test_days"])
    raise ValueError(mode)

def holdout_family_split(df, family, seed=42, test_size=0.2):
    """Train never sees `family`. Test = ALL rows of that family + benign from unseen blocks."""
    tr, te = block_split(df, seed, test_size)
    train = tr[(df.loc[tr, "family"] != family).values]
    benign_te = te[(df.loc[te, "family"] == "Benign").values]
    test = benign_te.union(df.index[df["family"] == family])
    return train, test
