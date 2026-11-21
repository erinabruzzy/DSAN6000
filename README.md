# DSAN6000 Big Data & Cloud Computing: Final Project
### Authors: Erin Brzusek and Jackson Howes

## Abstract 

New York City is one of the most recognizable cities in the world. The glitz and glamor of broadway theatres, the appeal of the incredible restaurant scene, and mindbending skyscrapers are contrasted with the likes of a homeless and housing insecure population not seen since the Great Depression:

> “In recent years, homelessness in New York City has reached the highest level since the Great Depression of the 1930s.” (via. *coalitionforthehomeless.org*)

Currently, New York City publishes ‘Stats & Reports’ on its over three-hundred homeless shelters to a public webpage daily. Unfortunately, these figures are static .pdfs that are overwritten daily and not conducive to modern data science techniques. The data is also scattered across multiple different reports, one describes ‘Pets Whose Owners Enter Homeless Shelters’ while another details ‘Encampment Cleanups and Aided Removals’. 

This project aims to create an interactive dashboard to house all of the data pertaining to the homeless population in the city. The mains tasks are to: 

- Preserve daily reports
- Capture historical trends
- Predict/forecast future needs
- Allow for the application of more modern analytical tools
- Simplify outputs for nontechnical audiences 


## Data Dashboard

The Streamlit dashboard separates the NYC DHS daily census totals from statistics
calculated over the synthetic intake records.

![](images/dashboard_1.png)

Intake records do not indicate
whether a person is currently in shelter, so their counts and age/children
statistics are labeled as intake-record measures rather than current occupancy.

![](images/dashboard_2.png)

## Python Environment (uv)

Dependencies are managed with [uv](https://docs.astral.sh/uv/) and pinned in
`uv.lock`. Spark itself needs Java 17 or newer on every node, and the PyPI
`pyspark` version (4.3.0) must match the Spark install in `~/spark`.

Install uv once on the Spark master:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

From the repository root, create `.venv` with the pinned Python (3.14) and
all dependencies, including the notebook tools:

```bash
uv sync
```

Run commands inside the environment with `uv run` (no activation needed). For
the EDA notebook, select the `.venv` kernel (`.venv/bin/python`) in VS Code or
Jupyter. After pulling changes that touch `pyproject.toml` or `uv.lock`, run
`uv sync` again.

## Running the Dashboard

Set the Spark standalone master URL and start the dashboard:

```bash
export SPARK_MASTER_URL="spark://<MASTER_PRIVATE_IP>:7077"
uv run streamlit run dashboard.py --server.address 127.0.0.1 --server.port 8501
```

The report path defaults to the current daily report object. Set
`DHS_DAILY_REPORT_PATH` if you upload a newer report under another name.

Open port 8501 through VS Code's port forwarding or an SSH tunnel; do not expose
the dashboard port publicly. `SPARK_MASTER_URL` defaults to `local[*]` if it is
not set. The dashboard reads the public S3 objects anonymously and caches the
daily report and the aggregated intake results for one hour.


## AWS S3 Bucket

In AWS Academy, open the lab’s AWS Details and copy its AWS CLI credentials/export block into the master terminal. 

On the master node terminal, open the credentials file:

First time:
```
mkdir -p ~/.aws
nano ~/.aws/credentials
```

Otherwise:
```
nano ~/.aws/credentials
```

In nano, paste the entire AWS Details block, including [default]:

```
[default]
aws_access_key_id = YOUR_ACCESS_KEY_ID
aws_secret_access_key = YOUR_SECRET_ACCESS_KEY
aws_session_token = YOUR_SESSION_TOKEN
```

Save with Ctrl+O, Enter, then exit with Ctrl+X. Keep the session token line if AWS Details provided one; temporary Academy credentials need it.

Then secure the file:

```
chmod 600 ~/.aws/credentials
```

Verify the credentials without printing them:

```
aws sts get-caller-identity
```

If that works, try an upload:
```
aws s3 cp \
  s3://dsan6000-final/nyc-shelters/intake_log.csv \
  s3://dsan6000-nyc-dashboard-data/source/nyc-shelters/intake_log.csv \
  --copy-props metadata-directive
```

**Public S3 Bucket Name: dsan6000-nyc-dashboard-data**

```
aws s3 ls s3://dsan6000-nyc-dashboard-data/ --recursive --no-sign-request
2026-11-20 16:31:18     221363 DHS_Daily_Report_20261120.csv
2026-11-20 16:26:36       3565 intakes_codebook.csv
2026-11-20 16:50:43 4055249999 source/nyc-shelters/intake_log.csv
```
