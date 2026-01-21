# n8n Production Setup for Mamba Stateful Training Pipeline

This directory contains a production-ready n8n setup that directly integrates with your Mamba training codebase.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         n8n ORCHESTRATION LAYER                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                         │
│  │   Trigger   │→ │   Config    │→ │   Split     │                         │
│  │ (Manual/    │  │   Setup     │  │  Symbols    │                         │
│  │  Schedule)  │  │             │  │  (Parallel) │                         │
│  └─────────────┘  └─────────────┘  └─────────────┘                         │
│                                           │                                 │
│                                           ▼                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    PER-SYMBOL PIPELINE (Sequential)                  │   │
│  │                                                                      │   │
│  │  ┌──────────────────┐    ┌──────────────────┐    ┌───────────────┐  │   │
│  │  │ 1. PREP_FAMILIES │ →  │ 2. STAGE_A       │ →  │ 3. STAGE_B    │  │   │
│  │  │                  │    │    SELECTOR      │    │    PHASE2     │  │   │
│  │  │ Generate feature │    │                  │    │               │  │   │
│  │  │ caches           │    │ Produce family   │    │ Train Mamba   │  │   │
│  │  │                  │    │ weights          │    │ model         │  │   │
│  │  └──────────────────┘    └──────────────────┘    └───────────────┘  │   │
│  │         │                        │                       │          │   │
│  │         ▼                        ▼                       ▼          │   │
│  │  cache/features/          artifacts/stage_a/     artifacts/optuna/  │   │
│  │  <SYM>_h<H>_merged.pq     <SYM>_h<H>/            <study>.db         │   │
│  │                           family_weights.json                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼  (docker exec)
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PYTHON WORKER CONTAINER                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  /mamba_workspace (volume mount)                                    │   │
│  │  ├── tools/prep_families.py                                         │   │
│  │  ├── tools/stage_a_selector.py                                      │   │
│  │  ├── tools/run_stage_b_stateful_phase2.py                           │   │
│  │  ├── cache/features/                                                │   │
│  │  └── artifacts/stage_a/, optuna_studies/                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Configure Environment

```bash
cd n8n
cp .env.example .env
# Edit .env with your settings (passwords, symbols, etc.)
nano .env
```

### 2. Start n8n Stack

```bash
./start_n8n.sh start
```

### 3. Fix Docker Socket Permissions (WSL2 only)

```bash
sudo chmod 666 /var/run/docker.sock
```

### 4. Access n8n Web UI

- **URL**: http://localhost:5678
- **User**: admin (or your `N8N_USER`)
- **Password**: mamba_admin_password (or your `N8N_PASSWORD`)

### 5. Import Workflow

```bash
./start_n8n.sh import
```

Or manually: **Settings → Import from File → `workflows/mamba_training_pipeline.json`**

## 📋 Pipeline Flow

The n8n workflow executes your 3-step training pipeline:

| Step | Entry Point | Input | Output |
|------|-------------|-------|--------|
| **1** | `dagster_prep_families` | EODHD data, GDELT | `cache/features/<SYM>_h<H>_merged.parquet` |
| **2** | `tools/stage_a_selector.py` | Merged parquet | `artifacts/stage_a/<SYM>_h<H>/family_weights_best.json` |
| **3** | `tools/run_stage_b_stateful_phase2.py` | Parquet + Weights | `artifacts/optuna_studies/<study>.db` |

### Data Flow Diagram

```
dagster_prep_families           stage_a_selector.py           run_stage_b_phase2.py
(dagster job execute)                  │                              │
      │                                ▼                              ▼
      ▼                         artifacts/stage_a/            artifacts/optuna_studies/
cache/features/                 ├─ AAPL_h63/                   ├─ phase2_v16_full_h63_AAPL.db
├─ AAPL_h63_merged.parquet      │   └─ family_weights.json     ├─ phase2_v16_full_h63_MSFT.db
├─ MSFT_h63_merged.parquet      ├─ MSFT_h63/                   └─ ..._best_trial.json
└─ NVDA_h63_merged.parquet      │   └─ family_weights.json
                                └─ NVDA_h63/
                                    └─ family_weights.json
```

### Dagster Command Used

```bash
# n8n runs this via docker exec:
DAGSTER_PREP_WF_START=2010-07-02 \
DAGSTER_PREP_WF_END=2025-07-01 \
DAGSTER_PREP_WRITE_MERGED=yes \
DAGSTER_PREP_MODE=stage-b \
python -m dagster job execute \
  -m dagster_prep_families.definitions \
  -j prep_families_job \
  --partition "symbol=AAPL|horizon=63"
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_SYMBOLS` | `AAPL,MSFT,NVDA...` | Comma-separated symbols to train |
| `DEFAULT_HORIZON` | `63` | Forecast horizon (trading days) |
| `WF_START` | `2010-07-02` | Walk-forward start date |
| `WF_END` | `2025-07-01` | Walk-forward end date |
| `WF_TRAIN_YEARS` | `5` | Training window size |
| `WF_STEP_DAYS` | `126` | Step size (6 months) |
| `OPTUNA_TRIALS` | `100` | Number of Optuna trials |
| `OPTUNA_TIMEOUT_HOURS` | `24` | Timeout per symbol |

### Schedule

Default: Daily at 2:00 AM UTC

To change, edit the **"Daily 2AM Trigger"** node in the workflow.

## � Containers

The Docker stack consists of 5 containers:

| Container | Image | Purpose |
|-----------|-------|---------|
| `mamba_n8n_main` | n8nio/n8n:latest | Main n8n UI & scheduler |
| `mamba_n8n_worker` | n8nio/n8n:latest | Worker for long-running tasks |
| `mamba_n8n_postgres` | postgres:16-alpine | Workflow & execution storage |
| `mamba_n8n_redis` | redis:7-alpine | Queue for worker communication |
| `mamba_python_worker` | python:3.12-slim | Executes Python pipeline scripts |

### Execution Pattern

n8n executes Python commands via docker exec:

```bash
docker exec mamba_python_worker bash -c "cd /mamba_workspace && python tools/prep_families.py ..."
```

## 🔧 Commands

```bash
# Start n8n
./start_n8n.sh start

# Stop n8n
./start_n8n.sh stop

# View logs
./start_n8n.sh logs

# Shell into n8n container
./start_n8n.sh shell

# Shell into Python worker
docker exec -it mamba_python_worker bash

# Import workflows
./start_n8n.sh import

# Check status
./start_n8n.sh status

# Fix docker socket (WSL2)
sudo chmod 666 /var/run/docker.sock

# Test Python execution
docker exec mamba_n8n_main docker exec mamba_python_worker python --version
```

## 📁 Directory Structure

```
n8n/
├── docker-compose.yml      # Docker Compose configuration (5 containers)
├── .env.example            # Example environment file
├── .env                    # Your environment config (gitignored)
├── start_n8n.sh            # Startup script
├── README.md               # This file
└── workflows/
    └── mamba_training_pipeline.json  # Main workflow template

# Created artifact directories:
artifacts/
├── stage_a/                # Stage-A outputs per symbol
│   ├── AAPL_h63/
│   │   └── family_weights_best.json
│   └── ...
├── optuna_studies/         # Optuna study DBs and best trials
│   ├── phase2_v16_full_h63_AAPL__label_base.db
│   └── phase2_v16_full_h63_AAPL__label_base_best_trial.json
└── backtests/              # Backtest results

logs/                       # Pipeline execution logs
```

## 🔗 Alternative: Native Shell Script

For simpler execution without n8n, use the bash script:

```bash
# Make executable
chmod +x tools/run_training_pipeline.sh

# Run for specific symbols
./tools/run_training_pipeline.sh AAPL,MSFT,NVDA 63 100

# Arguments: SYMBOLS HORIZON OPTUNA_TRIALS
```

This runs the same 3-script pipeline sequentially per symbol with logging to `logs/pipeline_<RUN_ID>.log`.

## 📊 Notifications (Optional)

Add notification nodes after the "Format Report" node:

- **Slack**: Use the Slack node with your webhook URL
- **Email**: Use the Send Email node with SMTP settings
- **Discord**: Use the Discord node with your webhook

## 🐛 Troubleshooting

### Docker not running / Permission denied
```bash
# Start docker
sudo systemctl start docker

# Fix socket permissions (WSL2)
sudo chmod 666 /var/run/docker.sock
```

### Docker pull fails with IPv6 errors (WSL2)
This is a known WSL2 issue. Apply the fix:
```bash
# Create docker systemd override
sudo mkdir -p /etc/systemd/system/docker.service.d
echo '[Service]
Environment="GODEBUG=netdns=cgo"' | sudo tee /etc/systemd/system/docker.service.d/override.conf

# Restart docker
sudo systemctl daemon-reload
sudo systemctl restart docker
```

### n8n can't execute docker commands
```bash
# Verify docker is mounted into container
docker exec mamba_n8n_main which docker
# Should show: /usr/bin/docker

# Verify socket is accessible
docker exec mamba_n8n_main docker ps
```

### Python command fails
```bash
# Test python-worker is running
docker exec mamba_python_worker python --version

# Test full chain from n8n
docker exec mamba_n8n_main docker exec mamba_python_worker python --version
```

### View container logs
```bash
# n8n main logs
docker logs mamba_n8n_main

# Python worker logs
docker logs mamba_python_worker

# All containers
docker compose -f n8n/docker-compose.yml logs -f
```

### Reset everything
```bash
cd n8n
docker compose down -v  # Remove containers AND volumes
docker compose up -d    # Fresh start
./start_n8n.sh import   # Re-import workflow
```

## 📈 Extending the Workflow

Common extensions:
1. **Add Slack alerts** on training completion
2. **Add error handling** with retry logic
3. **Add S3 upload** for prediction tapes
4. **Add Grafana** for metrics visualization
5. **Add Stage C** policy optimization step
