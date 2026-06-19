import pandas as pd
import requests
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

# Kenneth French factors
base_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
files = {
    "factors_daily": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "factors_monthly": "F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "momentum_daily": "F-F_Momentum_Factor_daily_CSV.zip",
    "momentum_monthly": "F-F_Momentum_Factor_CSV.zip",
}

# use a browser-like agent for French's server
headers = {"User-Agent": "Mozilla/5.0 (research; factor data download)"}

# set a common start for all factors
common_start = "1963-07-01"

data_dir = Path(__file__).resolve().parent.parent / "data"

# fetch a French zip and return its CSV text
def _download_csv(filename):
    response = requests.get(base_url + filename, headers=headers, timeout=60)
    response.raise_for_status()
    archive = ZipFile(BytesIO(response.content))
    return archive.read(archive.namelist()[0]).decode("latin-1")

# parse a French CSV into a decimal returns frame indexed by date
def _parse(text, frequency):
    lines = text.splitlines()

    header_row = None
    for i, line in enumerate(lines):
        if "Mkt-RF" in line or ("Mom" in line and "," in line):
            header_row = i
            break
    if header_row is None:
        raise ValueError("header row not found")

    columns = [c.strip() for c in lines[header_row].split(",")]
    columns[0] = "date"
    date_length = 8 if frequency == "daily" else 6

    records = []
    for line in lines[header_row + 1:]:
        parts = [p.strip() for p in line.split(",")]
        key = parts[0]
        if not key.isdigit() or len(key) != date_length:
            if records:  # stop at the appended annual block
                break
            continue
        records.append(parts)

    frame = pd.DataFrame(records, columns=columns).set_index("date")
    frame = frame.apply(pd.to_numeric, errors="coerce")

    if frequency == "daily":
        frame.index = pd.to_datetime(frame.index, format="%Y%m%d")
    else:
        frame.index = pd.to_datetime(frame.index, format="%Y%m").to_period("M").to_timestamp("M")

    return frame / 100.0

# combine the six factor columns from the 5-factor and momentum sets
def _factor_frame(five_factor, momentum):
    combined = pd.DataFrame(index=five_factor.index)
    combined["Mkt"] = five_factor["Mkt-RF"]
    combined["SMB"] = five_factor["SMB"]
    combined["HML"] = five_factor["HML"]
    combined["RMW"] = five_factor["RMW"]
    combined["CMA"] = five_factor["CMA"]

    mom_column = "Mom" if "Mom" in momentum.columns else momentum.columns[0]
    combined["Mom"] = momentum[mom_column].reindex(five_factor.index)
    return combined

# download, align and trim daily and monthly factor returns to a common window
def load_factor_data(start=common_start):
    daily_path = data_dir / "daily.csv"
    monthly_path = data_dir / "monthly.csv"

    # read the cache if it exists, otherwise download and save
    if daily_path.exists() and monthly_path.exists():
        daily = pd.read_csv(daily_path, index_col=0, parse_dates=True)
        monthly = pd.read_csv(monthly_path, index_col=0, parse_dates=True)
        return daily, monthly

    daily = _factor_frame(
        _parse(_download_csv(files["factors_daily"]), "daily"),
        _parse(_download_csv(files["momentum_daily"]), "daily"),
    )
    monthly = _factor_frame(
        _parse(_download_csv(files["factors_monthly"]), "monthly"),
        _parse(_download_csv(files["momentum_monthly"]), "monthly"),
    )

    daily = daily.loc[start:].dropna()
    monthly = monthly.loc[start:].dropna()

    data_dir.mkdir(exist_ok=True)
    daily.to_csv(daily_path)
    monthly.to_csv(monthly_path)
    return daily, monthly

if __name__ == "__main__":
    daily_returns, monthly_returns = load_factor_data()
    print("daily", daily_returns.shape)
    print("monthly", monthly_returns.shape)
    print(monthly_returns.head())