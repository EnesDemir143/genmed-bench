# Data Version Control (DVC) Guide

> How to use DVC for versioning large datasets with Google Drive remote storage.

---

## Overview

DVC (Data Version Control) tracks large files (LMDB databases, model checkpoints) that shouldn't be stored in Git. GenMed-Bench uses Google Drive as the remote storage backend.

---

## Prerequisites

### 1. Install DVC

```bash
# Already included in project dependencies
uv sync

# Or install separately
pip install dvc dvc-gdrive
```

### 2. Google Drive Desktop App (Recommended)

For faster uploads/downloads, install Google Drive Desktop:

1. **Download**: https://www.google.com/drive/download/
2. **Install** the desktop application
3. **Sign in** with your Google account
4. **Enable offline access** for the shared folder

This creates a local mount (e.g., `/Users/username/Google Drive/`) that syncs automatically.

---

## Initial Setup

### 1. Initialize DVC

```bash
# Already done in this project
dvc init
```

### 2. Add Google Drive Remote

```bash
# Create a folder in Google Drive, get folder ID from URL
# URL: https://drive.google.com/drive/folders/<FOLDER_ID>

dvc remote add -d gdrive gdrive://<FOLDER_ID>

# Example
dvc remote add -d gdrive gdrive://1ABC123xyz456
```

### 3. Configure (Optional)

```bash
# Use service account (for automation)
dvc remote modify gdrive gdrive_use_service_account true
dvc remote modify gdrive gdrive_service_account_json_file_path path/to/credentials.json

# Or use OAuth (interactive, one-time auth)
dvc remote modify gdrive gdrive_acknowledge_abuse_of_files true
```

---

## Daily Usage

### Pulling Data (Clone & Setup)

```bash
# Clone repository
git clone https://github.com/EnesDemir143/genmed-bench.git
cd genmed-bench

# Install dependencies
uv sync

# Pull data from remote
dvc pull
```

### Pushing Data (After Processing)

```bash
# Track new processed data
dvc add data/processed/nih/nih.lmdb

# Commit DVC file
git add data/processed/nih/nih.lmdb.dvc .gitignore
git commit -m "Add NIH LMDB database"

# Push to remote
dvc push
```

---

## Pipeline Management

### Current Pipeline

```yaml
# dvc.yaml
stages:
  prepare_nih:
    cmd: python scripts/prepare_data.py --dataset nih --image-size 256 --quality 95
    deps:
      - scripts/prepare_data.py
      - data/raw/chest-datasets/nih
    outs:
      - data/processed/nih/nih.lmdb

  prepare_vinbigdata:
    cmd: python scripts/prepare_data.py --dataset vinbigdata --image-size 256 --quality 95
    deps:
      - scripts/prepare_data.py
      - data/raw/chest-datasets/vinbigdata
    outs:
      - data/processed/vinbigdata/vinbigdata.lmdb
```

### Running Pipeline

```bash
# Run all stages
dvc repro

# Run specific stage
dvc repro prepare_nih

# Check status
dvc status
```

---

## Commands Reference

| Command | Description |
|---------|-------------|
| `dvc pull` | Download data from remote |
| `dvc push` | Upload data to remote |
| `dvc add <path>` | Track a file/folder |
| `dvc remove <path>.dvc` | Stop tracking |
| `dvc repro` | Run pipeline |
| `dvc status` | Check what's changed |
| `dvc diff` | Show differences |
| `dvc gc` | Clean unused cache |

---

## Google Drive Setup (Detailed)

### Option 1: OAuth Authentication (Easy)

First time you run `dvc pull` or `dvc push`:

1. Browser opens automatically
2. Sign in to Google account
3. Grant access to DVC
4. Token saved locally (~/.config/dvc/)

### Option 2: Service Account (Automation)

For CI/CD or server environments:

1. **Create Service Account**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create project → IAM & Admin → Service Accounts
   - Create service account, download JSON key

2. **Share Drive Folder**:
   - Share your Drive folder with service account email
   - Grant "Editor" access

3. **Configure DVC**:
   ```bash
   dvc remote modify gdrive gdrive_use_service_account true
   dvc remote modify gdrive gdrive_service_account_json_file_path credentials.json
   ```

### Option 3: Desktop App (Fastest)

1. Install [Google Drive Desktop](https://www.google.com/drive/download/)
2. Sign in and sync
3. Mount appears at:
   - macOS: `/Users/<username>/Google Drive/`
   - Windows: `G:\` or `My Drive`
   - Linux: via `google-drive-ocamlfuse`

4. Use local remote:
   ```bash
   dvc remote add -d local_gdrive "/Users/username/Google Drive/genmed-bench-data"
   ```

---

## Workflow Example

### Adding a New Dataset

```bash
# 1. Convert to LMDB
uv run python -m src.data.converters.new_dataset_converter

# 2. Track with DVC
dvc add data/processed/new_dataset/new_dataset.lmdb

# 3. Commit
git add data/processed/new_dataset/new_dataset.lmdb.dvc
git add .gitignore
git commit -m "Add new_dataset LMDB"

# 4. Push to remote
dvc push

# 5. Push git
git push
```

### Collaborator Workflow

```bash
# 1. Pull latest code
git pull

# 2. Pull latest data
dvc pull

# 3. Work on project...

# 4. If you modified data
dvc add <modified-files>
git add *.dvc
git commit -m "Update data"
dvc push
git push
```

---

## Troubleshooting

### Authentication Issues

```bash
# Clear cached credentials
rm -rf ~/.config/dvc/

# Re-authenticate
dvc pull
```

### Slow Transfers

- Use Google Drive Desktop for large files
- Check internet connection
- Consider using local cache: `dvc cache dir /path/to/fast/ssd`

### "Modified" Files After Pull

```bash
# Re-checkout files
dvc checkout

# If still issues
dvc fetch
dvc checkout --force
```

### Quota Exceeded

- Google Drive free: 15GB
- Consider Google Workspace for more storage
- Clean old versions: `dvc gc --workspace`

---

## Project Structure

```
genmed-bench/
├── .dvc/
│   ├── config          # Remote configuration
│   └── .gitignore
├── dvc.yaml            # Pipeline definition
├── dvc.lock            # Locked versions
├── data/
│   ├── processed/
│   │   ├── nih/
│   │   │   └── nih.lmdb.dvc      # DVC pointer file
│   │   └── vinbigdata/
│   │       └── vinbigdata.lmdb.dvc
│   └── .gitignore      # Ignores actual data
└── .gitignore
```

---

## Best Practices

1. **Always pull before work**: `dvc pull`
2. **Track processed data, not raw**: LMDB files, not original images
3. **Use .dvc files in git**: They're small pointer files
4. **Run pipeline with dvc repro**: Ensures reproducibility
5. **Clean cache periodically**: `dvc gc --workspace`
