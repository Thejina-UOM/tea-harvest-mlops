# Tea Flush Planner — tea harvest-readiness prediction (run and deploy)

Applied Machine Learning final assignment, MSc in AI, University of Moratuwa — Thejina (269106K)

An Explainable Boosting Machine predicts whether a Sri Lankan tea region is harvest-ready in a
given month, served through a Streamlit app and a FastAPI service with an MLOps pipeline
(DVC + MLflow on DagsHub, GitHub Actions, Docker, Azure Container Apps).

## Folder contents

| Path | Purpose |
|---|---|
| `notebooks/Tea_Harvest_Readiness_EBM.ipynb` | Full analysis; re-trains and saves the model and metrics |
| `frontend/` | Streamlit app |
| `app/` | FastAPI service (`/predict`, `/explain`, `/feedback`, `/health`) |
| `src/` | Shared features, data contract, DVC pipeline stages, quality gate, model registry |
| `models/tea_harvest_ebm.joblib` | Trained model used by the app |
| `reports/metrics`, `reports/figures` | Results the app's comparison and explanation pages display |
| `data/` | Dataset and the (empty) production feedback log |
| `monitoring/`, `replayer/` | Drift check and simulated production traffic |
| `infra/`, `.github/workflows/`, `Dockerfile*` | Azure resources, CI/CD, container images |
| `tests/` | Unit tests |

## Run locally (Windows, macOS, Linux — Python 3.11–3.13, 3.12 recommended)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt

streamlit run frontend/streamlit_app.py          # the app  → http://localhost:8501
uvicorn app.main:app --port 8000                 # the API (needs models/model.joblib from `dvc repro`,
                                                 #   or set MODEL_PATH=models/tea_harvest_ebm.joblib)
pytest                                           # 20 tests
```

Re-run the analysis: open `notebooks/Tea_Harvest_Readiness_EBM.ipynb` and *Restart & Run All*
(≈10–15 min; set `FAST_MODE = True` in the first code cell for ≈4 min).

## Deploy (GitHub Codespace, same flow as the MLOps workshop)

1. Push this folder to a **public** GitHub repo `tea-harvest-mlops`; connect it on DagsHub and create a token.
2. GitHub → Settings → Secrets and variables → Actions: secret `DAGSHUB_TOKEN`; variables
   `DAGSHUB_OWNER`, `DAGSHUB_REPO=tea-harvest-mlops`. Add the same three as Codespaces secrets.
3. In a Codespace:
   ```bash
   make doctor && make test
   uv run dvc init
   uv run dvc remote add -d origin https://dagshub.com/$DAGSHUB_OWNER/$DAGSHUB_REPO.dvc
   make dvc-auth
   uv run dvc add data/production/labelled.csv
   make experiments                         # all models logged to MLflow on DagsHub
   make repro && uv run dvc push            # prepare → train → evaluate
   uv run python -m src.gate
   uv run python -m src.registry promote    # first @champion
   git add . && git commit -m "Pipeline outputs" && git push
   ```
4. The push to `main` runs **CD → build** (images on GHCR). Make both packages public.
5. Azure Cloud Shell:
   `bash infra/bootstrap.sh <user>/tea-harvest-mlops ghcr.io/<user>/tea-harvest-mlops:latest`
   → add the 4 printed values as Actions variables, create environment `production` (reviewer = you).
6. *Actions → CD → Run workflow* → approve → canary 10 % → 100 %; the app URL is printed in the summary.
7. Drift drill: `uv run python replayer/replay.py <api-url> --mode climate --rows 600`, then *Actions → Monitor*.
8. After the demo: `az group delete --name rg-tea-mlops --yes`.

Free alternative for the app only: Streamlit Community Cloud → main file `frontend/streamlit_app.py`.

CI demo: pipeline checked on every pull request.
