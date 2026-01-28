import pandas as pd

def top_percent_cutoff(df, p=0.01):
    freq = (
        df.groupby(["sample", "templates"])
        .size()
        .reset_index(name="count")
        .sort_values(by=["sample", "templates"], ascending=True)
    )
    
    freq["cummulative"] = freq.groupby("sample")["count"].cumsum()
    freq["total"] = freq.groupby("sample")["count"].transform("sum")

    freq["cummulative_perc"] = freq["cummulative"] / freq["total"]
    freq.reset_index(drop=True, inplace=True)
    
    df = df.merge(freq[["sample", "templates", "cummulative_perc"]], on=["sample", "templates"], how='left')
    df_filtered = df[df["cummulative_perc"] >= 1-p]
    return df_filtered


def frequency_cutoff(df, min_freq=0.00001):
    df["total_templates"] = df.groupby("sample")["templates"].transform("sum")
    df["template_perc"] = df["templates"] / df["total_templates"]
    df_filtered = df[df["template_perc"] >= min_freq]
    return df_filtered
