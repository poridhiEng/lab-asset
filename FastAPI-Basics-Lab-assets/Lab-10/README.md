# **Lab 10 – GitHub Actions CI/CD for FastAPI**

Welcome to Lab 10! Now that you have a production-ready, load-balanced FastAPI application, it's time to automate testing, linting, and deployment using **GitHub Actions**. This lab turns your manual deployment process into a fully automated CI/CD pipeline.

## **Objectives**

By the end of this lab, you will:

- Create a GitHub Actions workflow that automatically runs tests on every commit
- Implement code linting with `flake8` to enforce code quality
- Build and push Docker images to Docker Hub automatically
- Deploy your application to a cloud server (EC2) via GitHub Actions
- Understand CI/CD best practices for Python/FastAPI applications

## **Why CI/CD Matters**

### **The Problem: Manual Deployment is Error-Prone**

Before CI/CD, here's what deployment looks like:

```bash
# Manual process - prone to errors
$ python -m pytest              # Did you remember to run tests?
$ flake8 app/                   # Did you lint?
$ docker build -t myapp .       # Did you tag correctly?
$ docker push myapp:latest      # Did you push to right registry?
$ ssh ec2 "docker pull..."      # Did you deploy?
```

**Risks:**
- Forgetting to run tests → broken code in production
- Inconsistent linting → messy codebase
- Manual image builds → version mismatches
- SSH deployment → security risks, no audit trail

### **The Solution: Automated CI/CD**

**CI (Continuous Integration):** Automatically test & validate code on every push

**CD (Continuous Deployment):** Automatically deploy validated code to production

```mermaid
graph LR
    subgraph "GitHub Actions CI/CD Pipeline"
        Push["Git Push"] --> Action["GitHub Actions Triggered"]
        Action --> Test["Run pytest"]
        Action --> Lint["Run flake8"]
        Test --> Build["Build Docker Image"]
        Lint --> Build
        Build --> Push["Push to Docker Hub"]
        Push --> Deploy["Deploy to EC2"]
    end
    
    style Action fill:#24292e,stroke:#24292e,stroke-width:2px,color:#fff
    style Test fill:#28a745,stroke:#28a745,stroke-width:2px,color:#fff
    style Lint fill:#0366d6,stroke:#0366d6,stroke-width:2px,color:#fff
    style Build fill:#f66a0a,stroke:#f66a0a,stroke-width:2px,color:#fff
    style Deploy fill:#6f42c1,stroke:#6f42c1,stroke-width:2px,color:#fff
```

**Benefits:**
-  **Consistency**: Every deployment follows the same steps
-  **Speed**: Deployments complete in 2-3 minutes, not 20-30 minutes
-  **Safety**: Tests must pass before deployment
-  **Audit trail**: Every deployment is logged in GitHub
-  **Rollback**: Easy to revert to previous version

---

## **Project Structure**

```
Lab-10/
├── .github/
│   └── workflows/
│       └── ci.yml          # Main CI/CD pipeline
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── test_main.py        # Tests for FastAPI
├── requirements.txt
├── requirements-dev.txt    # Dev dependencies (pytest, flake8)
├── Dockerfile
├── docker-compose.yml
└── .dockerignore
```

---

## **Step-by-Step Implementation Guide**

### **Step 1: Create Project Structure**

First, let's create the directory structure for our CI/CD project.

```bash
# Create app directory
mkdir -p app

# Create GitHub workflows directory
mkdir -p .github/workflows
```

**Verify the structure:**
```bash
ls -la
```

**Expected output:**
```
drwxr-xr-x  app/
drwxr-xr-x  .github/
```

### **Step 2: Create the FastAPI Application**

We'll create a FastAPI app similar to Lab 9, but optimized for CI/CD with proper test endpoints.

```bash
touch app/__init__.py
```

**`app/__init__.py`:**
```python
"""FastAPI application package."""

```

**What this does:**
- Makes `app` a Python package
- Allows importing modules from the app directory
- Required for pytest to discover tests

```bash
touch app/main.py
```

**`app/main.py`**
```python
"""FastAPI CI/CD Pipeline Lab."""
from fastapi import FastAPI, Request
import os
import socket

app = FastAPI(title="CI/CD Pipeline Lab")

INSTANCE_ID = os.getenv("INSTANCE_ID", "unknown")
HOSTNAME = socket.gethostname()


@app.get("/")
async def root(request: Request):
    """Return instance information."""
    return {
        "message": f"Hello from {INSTANCE_ID}!",
        "instance_id": INSTANCE_ID,
        "hostname": HOSTNAME,
        "client_ip": request.headers.get("X-Real-IP", request.client.host),
    }


@app.get("/health")
async def health():
    """Return health status."""
    return {
        "status": "healthy",
        "instance_id": INSTANCE_ID,
        "hostname": HOSTNAME
    }


@app.get("/api/data")
async def get_data():
    """Return sample data."""
    return {
        "data": [
            {"id": 1, "value": "Item 1"},
            {"id": 2, "value": "Item 2"},
            {"id": 3, "value": "Item 3"}
        ],
        "served_by": INSTANCE_ID,
        "hostname": HOSTNAME
    }

```

**Why these endpoints?**
- `/`: Main endpoint with instance information (for load balancing)
- `/health`: Health check endpoint (used by Docker and deployment verification)
- `/api/data`: Sample data endpoint (demonstrates API functionality)

**Verify the file was created:**
```bash
cat app/main.py | head -20
```

You should see the imports and FastAPI initialization.

---

## **Step 3: Create Test Files**

Tests are **critical** for CI/CD - they prevent broken code from being deployed. We'll use pytest with FastAPI's TestClient.

```bash
touch app/test_main.py
```

**`app/test_main.py`**
```python
"""Tests for FastAPI application."""
from fastapi.testclient import TestClient
from .main import app

client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint returns correct structure."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "instance_id" in data
    assert "hostname" in data


def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "instance_id" in data


def test_api_data_endpoint():
    """Test the data endpoint returns correct structure."""
    response = client.get("/api/data")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert isinstance(data["data"], list)
    assert len(data["data"]) == 3
    assert "served_by" in data


def test_missing_endpoint():
    """Test 404 error handling."""
    response = client.get("/nonexistent")
    assert response.status_code == 404

```

**What these tests do:**
1. **test_root_endpoint()**: Verifies the main endpoint returns correct JSON structure
2. **test_health_endpoint()**: Ensures health check works (critical for deployments)
3. **test_api_data_endpoint()**: Tests data endpoint returns list of 3 items
4. **test_missing_endpoint()**: Confirms 404 errors work correctly

**Verify tests are valid Python:**
```bash
python -m py_compile app/test_main.py
```

No output means the syntax is correct!

---

## **Step 4: Update Requirements Files**

We need two requirement files: one for production and one for development/testing.

```bash
touch requirements.txt requirements-dev.txt
```

**`requirements.txt`** (production dependencies)
```
fastapi==0.104.1
uvicorn==0.24.0
```

**`requirements-dev.txt`** (development dependencies)
```
-r requirements.txt

# Testing
pytest==7.4.3
pytest-cov==4.1.0
pytest-asyncio==0.21.1
httpx==0.25.2

# Linting
flake8==6.1.0
flake8-docstrings==1.7.0
black==23.11.0

# Type checking
mypy==1.7.1
```

**Why separate requirement files?**
- **requirements.txt**: Minimal dependencies for production Docker images (smaller, faster)
- **requirements-dev.txt**: Includes testing/linting tools (only needed during development/CI)

**Configuration explained:**
- `-r requirements.txt`: Includes all production dependencies
- **pytest**: Test framework
- **pytest-cov**: Code coverage reporting
- **pytest-asyncio**: Support for async test functions
- **httpx**: Required by TestClient for making HTTP requests in tests
- **flake8**: Python linter for code quality
- **flake8-docstrings**: Checks for docstring presence
- **black**: Code formatter (optional but recommended)
- **mypy**: Static type checker

**Verify files were created:**
```bash
ls -la requirements*.txt
```

---

## **Step 5: Create Dockerfile**

The Dockerfile defines how our application will be built and run in containers.

```bash
touch Dockerfile
```

**`Dockerfile`**
```dockerfile
# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Dockerfile configuration explained:**
- **FROM python:3.11-slim**: Uses minimal Python base image (smaller size)
- **WORKDIR /app**: Sets working directory inside container
- **RUN apt-get install gcc**: Installs compiler for some Python packages
- **--no-cache-dir**: Reduces image size by not caching pip downloads
- **EXPOSE 8000**: Documents which port the app uses
- **HEALTHCHECK**: Allows Docker/Kubernetes to verify app is running
  - Checks `/health` endpoint every 30 seconds
  - Retries 3 times before marking unhealthy
- **CMD**: Runs uvicorn server on container startup

**Verify Dockerfile syntax:**
```bash
cat Dockerfile
```

---

## **Step 6: Create GitHub Actions Workflow**

This is the **heart of the CI/CD pipeline**. The workflow file defines what happens when you push code.

```bash
touch .github/workflows/ci.yml
```

**`.github/workflows/ci.yml`**
```yaml
name: CI/CD Pipeline for FastAPI

on:
  # Trigger on push to main branch
  push:
    branches: [ main ]
    paths-ignore:
      - '**.md'  # Ignore README changes
  
  # Trigger on pull requests
  pull_request:
    branches: [ main ]
  
  # Manual trigger
  workflow_dispatch:

env:
  # Docker registry
  REGISTRY: docker.io
  IMAGE_NAME: ${{ vars.DOCKERHUB_USERNAME }}/fastapi-app

jobs:
  # ==========================================
  # TEST JOB
  # ==========================================
  test:
    runs-on: ubuntu-latest
    name: Run Tests & Linting
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-dev.txt
      
      - name: Run linting (flake8)
        run: |
          echo "Running flake8 linting..."
          python -m flake8 app/ --count --select=E9,F63,F7,F82 --show-source --statistics
          python -m flake8 app/ --count --max-complexity=10 --max-line-length=88 --statistics
      
      - name: Run tests with pytest
        run: |
          echo "Running pytest..."
          python -m pytest app/test_main.py -v --cov=app --cov-report=xml --cov-report=term
      
      - name: Upload test coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          flags: unittests
          name: codecov-umbrella
          fail_ci_if_error: false

  # ==========================================
  # BUILD & PUSH JOB
  # ==========================================
  build-and-push:
    runs-on: ubuntu-latest
    name: Build & Push Docker Image
    
    # Only run if tests passed
    needs: test
    # Only push on main branch pushes (not PRs)
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Login to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ vars.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=sha,prefix={{branch}}-
            type=raw,value=latest,enable={{is_default_branch}}
      
      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          platforms: linux/amd64,linux/arm64

  # ==========================================
  # DEPLOY TO EC2 (BONUS)
  # ==========================================
  deploy:
    runs-on: ubuntu-latest
    name: Deploy to EC2
    
    # Only deploy after successful build
    needs: build-and-push
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Deploy to EC2 using SSH
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.EC2_HOST }}
          username: ${{ secrets.EC2_USERNAME }}
          key: ${{ secrets.EC2_SSH_KEY }}
          script: |
            echo "Pulling latest image..."
            docker pull ${{ env.IMAGE_NAME }}:latest
            
            echo "Stopping old container..."
            docker stop fastapi-app || true
            docker rm fastapi-app || true
            
            echo "Starting new container..."
            docker run -d \
              --name fastapi-app \
              --restart unless-stopped \
              -p 8000:8000 \
              -e INSTANCE_ID=EC2-Production \
              ${{ env.IMAGE_NAME }}:latest
            
            echo "Cleaning up old images..."
            docker system prune -af

      - name: Verify deployment
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.EC2_HOST }}
          username: ${{ secrets.EC2_USERNAME }}
          key: ${{ secrets.EC2_SSH_KEY }}
          script: |
            sleep 10
            curl -f http://localhost:8000/health || exit 1
            echo "Deployment successful!"
```

**Workflow configuration explained:**

**Triggers:**
- `push: branches: [main]`: Runs on every push to main branch
- `paths-ignore: ['**.md']`: Skips workflow if only markdown files changed
- `pull_request`: Runs on PRs to main (for code review)
- `workflow_dispatch`: Allows manual trigger from GitHub UI

**Jobs structure:**
1. **test**: Runs tests and linting (always runs)
2. **build-and-push**: Builds Docker image (only if tests pass)
3. **deploy**: Deploys to EC2 (only if build succeeds)

**Test job details:**
- **runs-on: ubuntu-latest**: Uses GitHub's Ubuntu runner
- **actions/checkout@v4**: Checks out your code
- **actions/setup-python@v4**: Installs Python 3.11
- **cache: 'pip'**: Caches dependencies for faster builds
- **flake8 checks**:
  - First run: Critical errors only (E9, F63, F7, F82)
  - Second run: Code quality (complexity, line length)
- **pytest with coverage**: Runs tests and generates coverage report
- **codecov**: Uploads coverage to Codecov.io (optional)

**Build job details:**
- **needs: test**: Only runs if test job passes
- **if: github.ref == 'refs/heads/main'**: Only pushes images on main branch
- **docker/setup-buildx-action**: Enables advanced Docker features
- **docker/login-action**: Authenticates with Docker Hub
- **docker/metadata-action**: Generates image tags automatically
  - `latest` tag for main branch
  - SHA-based tags for versioning
- **docker/build-push-action**:
  - Builds for multiple platforms (amd64, arm64)
  - Uses GitHub Actions cache for faster builds

**Deploy job details:**
- **appleboy/ssh-action**: SSHs into EC2 and runs commands
- Pulls latest image from Docker Hub
- Stops old container gracefully
- Starts new container with environment variables
- Cleans up old images to save disk space
- Verifies deployment with health check

**Verify workflow file:**
```bash
cat .github/workflows/ci.yml | head -20
```

---

## **Step 7: Create Docker Compose for Local Testing**

Before pushing to GitHub, we should test everything locally. Docker Compose makes this easy.

```bash
touch docker-compose.yml
```

**`docker-compose.yml`**
```yaml
version: '3.8'

services:
  fastapi:
    build: .
    container_name: ci_cd_fastapi
    ports:
      - "8000:8000"
    environment:
      - INSTANCE_ID=Local-Dev
    volumes:
      - ./app:/app/app  # For hot-reloading during development
    restart: unless-stopped
    
    # Health check
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 5s
```

**Docker Compose configuration explained:**
- **build: .**: Builds image from Dockerfile in current directory
- **ports: "8000:8000"**: Maps host port 8000 to container port 8000
- **environment**: Sets INSTANCE_ID for testing
- **volumes**: Mounts local code for hot-reloading during development
- **restart: unless-stopped**: Auto-restart on failure
- **healthcheck**: Same as Dockerfile, verifies app is running

**Why use Docker Compose locally?**
- Tests the exact same Docker image that will be deployed
- Ensures Dockerfile works correctly
- Catches issues before pushing to GitHub
- Simulates production environment

---

## **Step 8: Create .dockerignore**

The `.dockerignore` file tells Docker which files to exclude when building images (similar to `.gitignore`).

```bash
touch .dockerignore
```

**`.dockerignore`**
```
.env
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.mypy_cache
.pytest_cache
.hypothesis
.github
README.md
.gitignore
Dockerfile*
docker-compose*
```

**Why exclude these files?**
- **.env, secrets**: Prevent leaking sensitive data into Docker images
- **__pycache__, *.pyc**: Python bytecode (regenerated anyway)
- **.git, .github**: Git history not needed in production image
- **.coverage, .pytest_cache**: Testing artifacts
- **README.md**: Documentation not needed at runtime
- **docker-compose***: Compose files for development only

**Benefits:**
- Smaller Docker images (faster builds, less storage)
- Better security (no secrets in images)
- Faster builds (less files to copy)

**Verify .dockerignore:**
```bash
cat .dockerignore
```

---

## **Step 9: Configure GitHub Secrets and Variables**

GitHub Secrets store sensitive data (passwords, tokens, SSH keys) securely. Variables store non-sensitive configuration.

**Why secrets are important:**
- Never hardcode credentials in code
- Secrets are encrypted by GitHub
- Only visible to authorized workflow runs
- Each repository has its own secrets

### **Part A: Set Up Docker Hub Access Token**

Before configuring GitHub, you need a Docker Hub account and access token.

**1. Create Docker Hub account** (if you don't have one):
- Go to https://hub.docker.com/signup
- Sign up with email and password
- Verify your email

**2. Create Docker Hub access token:**
- Log in to Docker Hub
- Click your username → **Account Settings**
- Go to **Security** → **Access Tokens**
- Click **New Access Token**
- Name: `github-actions-ci-cd`
- Permissions: **Read, Write, Delete**
- Click **Generate**
- **IMPORTANT**: Copy the token immediately (you won't see it again!)

Example token: `dckr_pat_abc123xyz456...`

### **Part B: Configure GitHub Repository Secrets**

Now we'll add the Docker Hub token to GitHub.

**1. Navigate to your GitHub repository:**
- Go to https://github.com/YOUR_USERNAME/YOUR_REPO
- Click **Settings** tab
- In left sidebar, click **Secrets and variables** → **Actions**

**2. Add Docker Hub Token (Secret):**
- Click **New repository secret**
- Name: `DOCKERHUB_TOKEN`
- Secret: Paste your Docker Hub access token
- Click **Add secret**

**Why use secrets?**
- Tokens are encrypted and never exposed in logs
- Only GitHub Actions runners can access them
- Prevents unauthorized access to your Docker Hub

### **Part C: Configure GitHub Repository Variables**

Variables store non-sensitive configuration (like usernames).

**1. In the same Settings → Secrets and variables → Actions page:**
- Click the **Variables** tab
- Click **New repository variable**

**2. Add Docker Hub Username (Variable):**
- Name: `DOCKERHUB_USERNAME`
- Value: Your Docker Hub username (e.g., `johndoe`)
- Click **Add variable**

**Why use variables instead of secrets?**
- Username is not sensitive (public information)
- Makes it easier to reference in workflow files
- Can be viewed in workflow logs for debugging

### **Part D: Optional - EC2 Deployment Secrets (Bonus)**

If you want to deploy to an EC2 instance, add these additional secrets:

**1. EC2_HOST (Secret):**
- Name: `EC2_HOST`
- Value: Your EC2 public IP address or domain (e.g., `54.123.45.67`)

**2. EC2_USERNAME (Secret):**
- Name: `EC2_USERNAME`
- Value: SSH username (usually `ec2-user` for Amazon Linux, `ubuntu` for Ubuntu)

**3. EC2_SSH_KEY (Secret):**
- Name: `EC2_SSH_KEY`
- Value: **Full contents** of your private SSH key file

To get your SSH key contents:
```bash
# Display your private key
cat ~/.ssh/your-key.pem

# Copy the entire output including:
# -----BEGIN RSA PRIVATE KEY-----
# (key content)
# -----END RSA PRIVATE KEY-----
```

**Paste the entire key** (including BEGIN and END lines) as the secret value.

**Summary of secrets/variables needed:**

| Type | Name | Required? | Description |
|------|------|-----------|-------------|
| **Secret** | `DOCKERHUB_TOKEN` | ✅ Required | Docker Hub access token for pushing images |
| **Variable** | `DOCKERHUB_USERNAME` | ✅ Required | Your Docker Hub username |
| **Secret** | `EC2_HOST` | Optional | EC2 public IP (only if using deploy job) |
| **Secret** | `EC2_USERNAME` | Optional | EC2 SSH username (only if using deploy job) |
| **Secret** | `EC2_SSH_KEY` | Optional | EC2 private SSH key (only if using deploy job) |

**Verify secrets are configured:**
- Go to Settings → Secrets and variables → Actions
- You should see your secrets listed (values are hidden)
- Variables should show the actual values

**Important notes:**
- Secrets are encrypted and cannot be viewed after creation
- If you lose a secret, delete it and create a new one
- Never commit secrets to your Git repository
- Use `.env` files locally (and add to `.gitignore`)

---

## **Step 10: Test the CI/CD Pipeline Locally**

Before pushing to GitHub, **always test locally** to catch errors early. This mimics exactly what GitHub Actions will do.

### **1. Create Virtual Environment and Install Dependencies**

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Linux/Mac:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install development dependencies (includes testing tools)
pip install -r requirements-dev.txt
```

**Expected output:**
```
Successfully installed fastapi-0.104.1 uvicorn-0.24.0 pytest-7.4.3 ...
```

**Verify installation:**
```bash
python -m pytest --version
python -m flake8 --version
```

### **2. Run Linting (Code Quality Checks)**

Linting catches syntax errors, style issues, and code smells before tests run.

```bash
# Run flake8 - Critical errors only (E9, F63, F7, F82)
python -m flake8 app/ --count --select=E9,F63,F7,F82 --show-source --statistics
```

**Expected output (if no errors):**
```
0
```

**If you see errors:**
```
app/main.py:10:1: F821 undefined name 'something'
```
Fix them before proceeding!

```bash
# Run flake8 - Code quality checks
python -m flake8 app/ --count --max-complexity=10 --max-line-length=88 --statistics
```

**Expected output:**
```
app/main.py:154:89: E501 line too long (92 > 88 characters)
1     E501 line too long
```

**What these checks do:**
- **E9**: Runtime errors (syntax errors, undefined names)
- **F63**: Invalid syntax in f-strings
- **F7**: Syntax errors in statements
- **F82**: Undefined names in `__all__`
- **max-complexity=10**: Functions shouldn't be too complex
- **max-line-length=88**: Lines shouldn't exceed 88 characters (Black standard)

### **3. Run Tests with pytest**

Tests ensure your code works correctly before deployment.

```bash
# Run tests with verbose output and coverage
python -m pytest app/test_main.py -v --cov=app --cov-report=xml --cov-report=term
```

**Expected output:**
```
================================= test session starts =================================
platform linux -- Python 3.11.0, pytest-7.4.3, pluggy-1.0.0
collected 4 items

app/test_main.py::test_root_endpoint PASSED                                     [ 25%]
app/test_main.py::test_health_endpoint PASSED                                   [ 50%]
app/test_main.py::test_api_data_endpoint PASSED                                 [ 75%]
app/test_main.py::test_missing_endpoint PASSED                                  [100%]

---------- coverage: platform linux, python 3.11.0 -----------
Name                 Stmts   Miss  Cover
------------------------------------------
app/__init__.py          1      0   100%
app/main.py             28      0   100%
app/test_main.py        20      0   100%
------------------------------------------
TOTAL                   49      0   100%

================================== 4 passed in 0.12s ==================================
```

**What this means:**
- ✅ All 4 tests passed
- 100% code coverage (every line tested)
- Generated `coverage.xml` for codecov

**If tests fail:**
```
FAILED app/test_main.py::test_root_endpoint - AssertionError: assert 'messag' in {...}
```
Fix the issue in your code and re-run tests.

### **4. Build Docker Image Locally**

Test that your Dockerfile builds successfully.

```bash
# Build Docker image
docker build -t fastapi-ci-test .
```

**Expected output:**
```
[+] Building 15.3s (12/12) FINISHED
 => [internal] load build definition from Dockerfile
 => [1/6] FROM docker.io/library/python:3.11-slim
 => [2/6] WORKDIR /app
 => [3/6] RUN apt-get update && apt-get install -y gcc
 => [4/6] COPY requirements.txt .
 => [5/6] RUN pip install --no-cache-dir -r requirements.txt
 => [6/6] COPY app/ ./app/
 => exporting to image
 => => naming to docker.io/library/fastapi-ci-test
```

**Verify image was created:**
```bash
docker images | grep fastapi-ci-test
```

**Expected output:**
```
fastapi-ci-test   latest   a1b2c3d4e5f6   2 minutes ago   250MB
```

### **5. Run Container with Docker Compose**

Start the containerized application locally.

```bash
# Start container in detached mode
docker compose up -d

# Check container is running
docker compose ps
```

**Expected output:**
```
NAME            IMAGE                  STATUS         PORTS
ci_cd_fastapi   lab-10-fastapi         Up 5 seconds   0.0.0.0:8000->8000/tcp
```

**Check container health:**
```bash
docker inspect ci_cd_fastapi | grep -A 5 Health
```

### **6. Test Endpoints**

Verify the application responds correctly.

```bash
# Test root endpoint
curl http://localhost:8000/
```

**Expected output:**
```json
{
  "message": "Hello from Local-Dev!",
  "instance_id": "Local-Dev",
  "hostname": "a1b2c3d4e5f6",
  "client_ip": "172.17.0.1"
}
```

```bash
# Test health endpoint
curl http://localhost:8000/health
```

**Expected output:**
```json
{
  "status": "healthy",
  "instance_id": "Local-Dev",
  "hostname": "a1b2c3d4e5f6"
}
```

```bash
# Test data endpoint
curl http://localhost:8000/api/data
```

**Expected output:**
```json
{
  "data": [
    {"id": 1, "value": "Item 1"},
    {"id": 2, "value": "Item 2"},
    {"id": 3, "value": "Item 3"}
  ],
  "served_by": "Local-Dev",
  "hostname": "a1b2c3d4e5f6"
}
```

### **7. Check Container Logs**

View application logs to verify it's running correctly.

```bash
# View logs
docker logs ci_cd_fastapi

# Follow logs in real-time
docker logs -f ci_cd_fastapi
```

**Expected output:**
```
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### **8. Clean Up**

Stop and remove containers when testing is complete.

```bash
# Stop containers
docker compose down

# Remove test image (optional)
docker rmi fastapi-ci-test
```

**Summary of local testing:**
✅ Linting passed - Code quality is good
✅ Tests passed - Application works correctly
✅ Docker builds - Image creation successful
✅ Container runs - Application is healthy
✅ Endpoints respond - All routes working

**If everything works locally**, you're ready to push to GitHub!

---

## **Step 11: Push to GitHub and Watch the Magic**

### **1. Create GitHub Repository**

**If you don't have a repository yet:**

1. Go to https://github.com/new
2. Repository name: `fastapi-cicd-lab` (or any name you like)
3. Description: `FastAPI CI/CD Pipeline with GitHub Actions`
4. Choose **Public** or **Private**
5. **Do NOT** initialize with README, .gitignore, or license
6. Click **Create repository**

**Copy the repository URL:**
```
https://github.com/YOUR_USERNAME/fastapi-cicd-lab.git
```

### **2. Initialize Git Repository Locally**

```bash
# Initialize git repository
git init

# Check current branch name
git branch
```

**If not on `main` branch:**
```bash
# Rename to main (if on master)
git branch -M main
```

### **3. Add and Commit Files**

```bash
# Check which files will be committed
git status

# Add all files
git add .

# Verify files are staged
git status
```

**Expected output:**
```
Changes to be committed:
  new file:   .dockerignore
  new file:   .github/workflows/ci.yml
  new file:   Dockerfile
  new file:   app/__init__.py
  new file:   app/main.py
  new file:   app/test_main.py
  new file:   docker-compose.yml
  new file:   requirements-dev.txt
  new file:   requirements.txt
```

```bash
# Commit with descriptive message
git commit -m "feat: add CI/CD pipeline with GitHub Actions

- FastAPI app with health checks
- Pytest tests for all endpoints
- GitHub Actions workflow (test, build, deploy)
- Docker support with multi-stage builds
- Automated deployment to Docker Hub"
```

**Expected output:**
```
[main (root-commit) a1b2c3d] feat: add CI/CD pipeline with GitHub Actions
 9 files changed, 450 insertions(+)
 create mode 100644 .dockerignore
 ...
```

### **4. Connect to GitHub Remote**

```bash
# Add GitHub repository as remote
git remote add origin https://github.com/YOUR_USERNAME/fastapi-cicd-lab.git

# Verify remote was added
git remote -v
```

**Expected output:**
```
origin  https://github.com/YOUR_USERNAME/fastapi-cicd-lab.git (fetch)
origin  https://github.com/YOUR_USERNAME/fastapi-cicd-lab.git (push)
```

### **5. Push to GitHub (Triggers CI/CD!)**

```bash
# Push to main branch
git push -u origin main
```

**Expected output:**
```
Enumerating objects: 15, done.
Counting objects: 100% (15/15), done.
Delta compression using up to 8 threads
Compressing objects: 100% (12/12), done.
Writing objects: 100% (15/15), 5.23 KiB | 1.74 MiB/s, done.
Total 15 (delta 0), reused 0 (delta 0), pack-reused 0
To https://github.com/YOUR_USERNAME/fastapi-cicd-lab.git
 * [new branch]      main -> main
Branch 'main' set up to track remote branch 'main' from 'origin'.
```

**🎉 GitHub Actions is now triggered automatically!**

### **6. Monitor the Pipeline in Real-Time**

**Navigate to GitHub Actions:**
1. Go to your repository: `https://github.com/YOUR_USERNAME/fastapi-cicd-lab`
2. Click the **Actions** tab
3. You should see a workflow run in progress: `"feat: add CI/CD pipeline with GitHub Actions"`

**What you'll see:**
- ⏳ Yellow circle: Workflow is running
- ✅ Green checkmark: Job succeeded
- ❌ Red X: Job failed

**Click on the workflow run** to see details:

**Jobs overview:**
```
✅ test (Run Tests & Linting) - 2m 15s
✅ build-and-push (Build & Push Docker Image) - 3m 45s
⏳ deploy (Deploy to EC2) - Running...
```

**Click on a job** to see detailed logs:

**Example test job logs:**
```
Run actions/checkout@v4
Run actions/setup-python@v4
  Python version: 3.11.0
Run Install dependencies
  Successfully installed fastapi-0.104.1 pytest-7.4.3 ...
Run Run linting (flake8)
  Running flake8 linting...
  0 errors found
Run Run tests with pytest
  ================================ 4 passed in 0.15s ================================
  Coverage: 100%
```

**Example build job logs:**
```
Run docker/setup-buildx-action@v3
Run docker/login-action@v3
  Login Succeeded
Run docker/build-push-action@v5
  Building Docker image...
  #8 [2/6] WORKDIR /app
  #9 [3/6] RUN apt-get update && apt-get install -y gcc
  #10 [4/6] COPY requirements.txt .
  #11 [5/6] RUN pip install --no-cache-dir -r requirements.txt
  #12 [6/6] COPY app/ ./app/
  #13 exporting to image
  #13 pushing layers
  #13 pushing manifest for docker.io/YOUR_USERNAME/fastapi-app:latest
```

### **7. Verify Docker Image on Docker Hub**

After the build job completes:

1. Go to https://hub.docker.com/
2. Log in to your account
3. Navigate to **Repositories**
4. You should see `YOUR_USERNAME/fastapi-app`
5. Click on it to see tags:
   - `latest`
   - `main-abc123` (SHA-based tag)

**Check image details:**
- Size: ~250MB
- Last pushed: Just now
- Architecture: linux/amd64, linux/arm64

### **8. Common Issues and Troubleshooting**

**Issue 1: Workflow doesn't trigger**
- **Cause**: Workflow file not in correct location
- **Solution**: Ensure file is at `.github/workflows/ci.yml` (not `github/workflows/`)
- **Verify**: Run `ls -la .github/workflows/` and confirm `ci.yml` exists

**Issue 2: Tests fail on GitHub but pass locally**
```
FAILED app/test_main.py::test_root_endpoint
```
- **Cause**: Missing dependencies or environment differences
- **Solution**: Check that `requirements-dev.txt` includes all test dependencies
- **Debug**: Click on failed test in Actions logs to see exact error

**Issue 3: Docker build fails with "permission denied"**
```
ERROR: failed to solve: failed to compute cache key
```
- **Cause**: Docker login failed (incorrect DOCKERHUB_TOKEN)
- **Solution**:
  - Verify secret is set: Settings → Secrets → DOCKERHUB_TOKEN
  - Regenerate Docker Hub access token
  - Update secret in GitHub

**Issue 4: Linting fails with "line too long"**
```
E501 line too long (92 > 88 characters)
```
- **Cause**: Code doesn't meet style guidelines
- **Solution**: Break long lines or adjust `.flake8` config
- **Quick fix**:
  ```bash
  # Auto-format code with black
  black app/
  ```

**Issue 5: Coverage upload fails**
```
Error uploading to Codecov
```
- **Cause**: Codecov integration not configured (this is optional)
- **Solution**: Either:
  - Set up Codecov account and add CODECOV_TOKEN secret
  - Or set `fail_ci_if_error: false` (already configured in our workflow)

**Issue 6: Deploy job skipped**
```
deploy / Deploy to EC2 - Skipped
```
- **Cause**: EC2 secrets not configured (this is expected if you haven't set up EC2)
- **Solution**: This is normal! Deploy job only runs if EC2 secrets are configured

---

## **Step 12: Testing the Pipeline**

Now that your pipeline is running, let's test different scenarios to understand how CI/CD behaves.

### **Scenario 1: Successful Pipeline (All Tests Pass)**

This demonstrates the happy path - everything works correctly.

**1. Make a small code improvement:**

Edit `app/main.py` and add a new endpoint:
```python
@app.get("/version")
async def version():
    """Return API version."""
    return {"version": "1.0.0"}
```

**2. Commit and push:**
```bash
git add .
git commit -m "feat: add version endpoint"
git push origin main
```

**3. Watch the pipeline:**

Go to GitHub Actions → Your workflow

**Expected flow:**
```
   test (Run Tests & Linting) - PASS
   → Tests run (existing tests pass)
   → Linting passes
   → Takes ~2 minutes

   build-and-push (Build & Push Docker Image) - PASS
   → Docker image builds
   → Pushes to Docker Hub
   → Tags: latest, main-abc123
   → Takes ~3-4 minutes

   deploy (Deploy to EC2) - PASS (if EC2 configured)
   → Pulls new image
   → Deploys to server
   → Health check passes
   → Takes ~1-2 minutes
```

**Result:**  New code deployed to production in ~5-7 minutes automatically!

**4. Verify on Docker Hub:**
- Go to https://hub.docker.com/
- Check your repository
- New image should be there with latest timestamp

### **Scenario 2: Tests Fail (Pipeline Stops)**

This demonstrates how CI/CD prevents broken code from being deployed.

**1. Introduce a bug:**

Edit `app/test_main.py` and change a test:
```python
def test_root_endpoint():
    """Test the root endpoint returns correct structure."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "wrong_field" in data  # This will fail!
```

**2. Commit and push:**
```bash
git add .
git commit -m "bug: intentional test failure"
git push origin main
```

**3. Watch the pipeline fail:**

**Expected flow:**
```
❌ test (Run Tests & Linting) - FAIL
   → test_root_endpoint FAILED
   → AssertionError: assert 'wrong_field' in {...}
   → Takes ~2 minutes

⏭️ build-and-push - SKIPPED
   → Doesn't run because tests failed

⏭️ deploy - SKIPPED
   → Doesn't run because build was skipped
```

**Result:** ❌ Bad code blocked from production! **This is exactly what we want!**

**4. You'll receive notification:**
- GitHub sends email: "Workflow run failed"
- Pull requests would show red X
- Cannot merge until tests pass

**5. Fix the bug:**
```python
def test_root_endpoint():
    """Test the root endpoint returns correct structure."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data  # Fixed!
```

```bash
git add .
git commit -m "fix: correct test assertion"
git push origin main
```

Now the pipeline will succeed again!

### **Scenario 3: Linting Fails (Code Quality Issue)**

This demonstrates how linting enforces code quality standards.

**1. Add code with style issues:**

Edit `app/main.py` and add a very long line:
```python
@app.get("/long-line")
async def long_line():
    return {"message": "This is an extremely long line that violates the max line length of 88 characters and will be caught by flake8"}
```

**2. Commit and push:**
```bash
git add .
git commit -m "feat: add endpoint with long line"
git push origin main
```

**3. Watch linting fail:**

**Expected flow:**
```
❌ test (Run Tests & Linting) - FAIL
   → Linting:
     app/main.py:45:89: E501 line too long (150 > 88 characters)
   → 1 error found
   → Tests would have passed, but linting failed first!

⏭️ build-and-push - SKIPPED
⏭️ deploy - SKIPPED
```

**4. Fix the style issue:**
```python
@app.get("/long-line")
async def long_line():
    return {
        "message": (
            "This is a properly formatted line that follows "
            "the 88 character limit"
        )
    }
```

Or use auto-formatter:
```bash
black app/main.py
git add .
git commit -m "style: fix line length with black formatter"
git push origin main
```

### **Scenario 4: Pull Request Workflow (Code Review)**

This demonstrates how CI/CD works with pull requests for team collaboration.

**1. Create a feature branch:**
```bash
git checkout -b feature/add-status-endpoint
```

**2. Add a new feature:**

Edit `app/main.py`:
```python
@app.get("/status")
async def status():
    """Return system status."""
    return {
        "status": "operational",
        "uptime": "24h",
        "instance_id": INSTANCE_ID
    }
```

Add a test in `app/test_main.py`:
```python
def test_status_endpoint():
    """Test status endpoint."""
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
```

**3. Commit and push feature branch:**
```bash
git add .
git commit -m "feat: add status endpoint

- Returns system operational status
- Includes uptime and instance info
- Added test coverage"
git push origin feature/add-status-endpoint
```

**4. Create Pull Request on GitHub:**
- Go to your repository
- Click **Pull requests** → **New pull request**
- Base: `main` ← Compare: `feature/add-status-endpoint`
- Click **Create pull request**
- Add title and description

**5. Watch PR checks:**

GitHub automatically runs the workflow on the PR:

```
✅ test (Run Tests & Linting) - PASS
   → All tests pass (including new test)
   → Code quality approved

⏭️ build-and-push - SKIPPED
   → Doesn't run on PRs (only on main branch)

⏭️ deploy - SKIPPED
   → Doesn't run on PRs (only on main branch)
```

**Why only tests run on PRs:**
- Tests verify the code works
- But we don't deploy until code is merged to main
- Prevents accidental deployments from unreviewed code

**6. Merge the PR:**
- Review the code changes
- Check that tests passed (green checkmark)
- Click **Merge pull request**
- Click **Confirm merge**

**7. Now the full pipeline runs:**

After merging to main:
```
✅ test - PASS
✅ build-and-push - PASS (now runs!)
✅ deploy - PASS (now runs!)
```

**Result:** Feature deployed to production only after review and tests pass!

### **Scenario 5: Manual Workflow Trigger (workflow_dispatch)**

Sometimes you need to trigger the workflow manually without pushing code.

**1. Go to GitHub Actions:**
- Repository → **Actions** tab
- Select **CI/CD Pipeline for FastAPI** workflow
- Click **Run workflow** button (on the right side)
- Select branch: `main`
- Click **Run workflow**

**Use cases for manual trigger:**
- Redeploy without code changes
- Test deployment to production
- Debug workflow issues
- Force rebuild of Docker images

**Expected flow:**
Same as automatic trigger - full pipeline runs.

---

## **Step 13: Bonus - Advanced CI/CD Features**

These advanced patterns are used in production environments at scale. Implement them after mastering the basics!

### **1. Multi-Environment Deployment (Staging + Production)**

**Use case:** Test changes in a staging environment before deploying to production.

**How it works:**
- `develop` branch → deploys to staging server
- `main` branch → deploys to production server

**Implementation:**

**1. Add staging secrets to GitHub:**
```
STAGING_EC2_HOST
STAGING_EC2_USERNAME
STAGING_EC2_SSH_KEY
```

**2. Update `.github/workflows/ci.yml`:**

Replace the single deploy job with:

```yaml
deploy-staging:
  runs-on: ubuntu-latest
  name: Deploy to Staging
  needs: build-and-push
  if: github.event_name == 'push' && github.ref == 'refs/heads/develop'

  steps:
    - name: Deploy to Staging Server
      uses: appleboy/ssh-action@master
      with:
        host: ${{ secrets.STAGING_EC2_HOST }}
        username: ${{ secrets.STAGING_EC2_USERNAME }}
        key: ${{ secrets.STAGING_EC2_SSH_KEY }}
        script: |
          docker pull ${{ env.IMAGE_NAME }}:latest
          docker stop fastapi-staging || true
          docker rm fastapi-staging || true
          docker run -d \
            --name fastapi-staging \
            --restart unless-stopped \
            -p 8000:8000 \
            -e INSTANCE_ID=Staging \
            ${{ env.IMAGE_NAME }}:latest
          sleep 10
          curl -f http://localhost:8000/health || exit 1

deploy-production:
  runs-on: ubuntu-latest
  name: Deploy to Production
  needs: build-and-push
  if: github.event_name == 'push' && github.ref == 'refs/heads/main'

  steps:
    - name: Deploy to Production Server
      uses: appleboy/ssh-action@master
      with:
        host: ${{ secrets.PROD_EC2_HOST }}
        username: ${{ secrets.PROD_EC2_USERNAME }}
        key: ${{ secrets.PROD_EC2_SSH_KEY }}
        script: |
          docker pull ${{ env.IMAGE_NAME }}:latest
          docker stop fastapi-prod || true
          docker rm fastapi-prod || true
          docker run -d \
            --name fastapi-prod \
            --restart unless-stopped \
            -p 8000:8000 \
            -e INSTANCE_ID=Production \
            ${{ env.IMAGE_NAME }}:latest
          sleep 10
          curl -f http://localhost:8000/health || exit 1
```

**Workflow:**
```
develop branch → Staging server
main branch → Production server
```

**Benefit:** Test in staging before promoting to production!

### **2. Automated Rollback on Failed Deployment**

**Use case:** Automatically revert to previous version if new deployment fails health checks.

**How it works:**
1. Deploy new container with temporary name
2. Run health checks
3. If healthy: replace old container
4. If unhealthy: delete new container, keep old one running

**Implementation:**

Update the deploy job script:

```yaml
script: |
  echo "Pulling latest image..."
  docker pull ${{ env.IMAGE_NAME }}:latest

  echo "Starting new container with temporary name..."
  docker run -d \
    --name fastapi-app-new \
    --restart unless-stopped \
    -p 8001:8000 \
    -e INSTANCE_ID=EC2-Production \
    ${{ env.IMAGE_NAME }}:latest

  echo "Waiting for new container to be ready..."
  sleep 10

  echo "Running health check on new container..."
  if curl -f http://localhost:8001/health; then
    echo "✅ Health check passed! Switching to new container..."

    # Stop old container
    docker stop fastapi-app || true
    docker rm fastapi-app || true

    # Rename new container
    docker stop fastapi-app-new
    docker rm fastapi-app-new

    # Start with correct port mapping
    docker run -d \
      --name fastapi-app \
      --restart unless-stopped \
      -p 8000:8000 \
      -e INSTANCE_ID=EC2-Production \
      ${{ env.IMAGE_NAME }}:latest

    echo "✅ Deployment successful!"
  else
    echo "❌ Health check failed! Rolling back..."

    # Remove failed new container
    docker stop fastapi-app-new || true
    docker rm fastapi-app-new || true

    # Old container still running on port 8000
    echo "✅ Rollback complete. Old version still running."
    exit 1
  fi
```

**Benefit:** Zero downtime even if deployment fails!

### **3. Slack/Discord Notifications on Deployment**

**Use case:** Get notified in your team chat when deployments succeed or fail.

**Implementation for Slack:**

**1. Create Slack Webhook:**
- Go to https://api.slack.com/messaging/webhooks
- Click **Create New App**
- Choose "From scratch"
- App name: "GitHub Actions CI/CD"
- Select your workspace
- Go to **Incoming Webhooks** → **Add New Webhook to Workspace**
- Select channel: `#deployments`
- Copy the webhook URL

**2. Add secret to GitHub:**
```
SLACK_WEBHOOK = https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXX
```

**3. Update workflow:**

Add this step at the end of the deploy job:

```yaml
- name: Notify Slack on Success
  if: success()
  uses: 8398a7/action-slack@v3
  with:
    status: custom
    custom_payload: |
      {
        "text": "✅ Deployment Successful!",
        "attachments": [{
          "color": "good",
          "fields": [
            {
              "title": "Repository",
              "value": "${{ github.repository }}",
              "short": true
            },
            {
              "title": "Branch",
              "value": "${{ github.ref }}",
              "short": true
            },
            {
              "title": "Commit",
              "value": "${{ github.event.head_commit.message }}",
              "short": false
            },
            {
              "title": "Author",
              "value": "${{ github.actor }}",
              "short": true
            }
          ]
        }]
      }
  env:
    SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}

- name: Notify Slack on Failure
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: custom
    custom_payload: |
      {
        "text": "❌ Deployment Failed!",
        "attachments": [{
          "color": "danger",
          "fields": [
            {
              "title": "Repository",
              "value": "${{ github.repository }}",
              "short": true
            },
            {
              "title": "Branch",
              "value": "${{ github.ref }}",
              "short": true
            },
            {
              "title": "Failed Step",
              "value": "Check GitHub Actions logs",
              "short": false
            }
          ]
        }]
      }
  env:
    SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

**Slack messages will show:**
- ✅ Green message: Deployment succeeded
- ❌ Red message: Deployment failed
- Commit message, author, branch info

**Benefit:** Team stays informed without checking GitHub constantly!

### **4. Performance Testing in CI/CD**

**Use case:** Ensure your API meets performance requirements before deployment.

**Implementation:**

**1. Add performance test:**

Create `app/test_performance.py`:
```python
"""Performance tests."""
import time
from fastapi.testclient import TestClient
from .main import app

client = TestClient(app)


def test_response_time():
    """Ensure response time is under 100ms."""
    start = time.time()
    response = client.get("/")
    elapsed = (time.time() - start) * 1000  # Convert to ms

    assert response.status_code == 200
    assert elapsed < 100, f"Response time {elapsed}ms exceeds 100ms limit"


def test_concurrent_requests():
    """Test handling multiple concurrent requests."""
    import concurrent.futures

    def make_request():
        return client.get("/health").status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(50)]
        results = [f.result() for f in futures]

    # All requests should succeed
    assert all(status == 200 for status in results)
```

**2. Update workflow to run performance tests:**

```yaml
- name: Run performance tests
  run: |
    echo "Running performance tests..."
    python -m pytest app/test_performance.py -v
```

**Benefit:** Catch performance regressions before they reach production!

### **5. Automated Security Scanning**

**Use case:** Detect security vulnerabilities in dependencies before deployment.

**Implementation:**

Add to workflow:

```yaml
security-scan:
  runs-on: ubuntu-latest
  name: Security Scan

  steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'

    - name: Upload Trivy results to GitHub Security
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'
```

**What it does:**
- Scans for known vulnerabilities in dependencies
- Reports findings to GitHub Security tab
- Fails build if critical vulnerabilities found

**Benefit:** Prevents deploying code with known security issues!

## **Step 14: Cleanup and Best Practices**

### **Cleanup Commands**

**Clean up Docker Hub (save storage space):**

Old images accumulate over time. Clean them periodically.

```bash
# List all your Docker Hub tags
# Visit: https://hub.docker.com/repository/docker/YOUR_USERNAME/fastapi-app/tags

# Delete old tags using Docker Hub UI or CLI
docker rmi YOUR_USERNAME/fastapi-app:old-tag
```

**Docker Hub has limits:**
- Free tier: Unlimited public repositories
- Image pulls: 200 per 6 hours (unauthenticated)
- Consider upgrading if you exceed limits

**Clean up local Docker:**

```bash
# Remove unused images locally
docker image prune -a

# Remove stopped containers
docker container prune

# Remove all unused resources
docker system prune -a --volumes
```

**Verify cleanup:**
```bash
docker images
docker ps -a
```

### **EC2 Server Maintenance (If Using Deploy Job)**

**Connect to your EC2 instance:**
```bash
ssh -i ~/.ssh/your-key.pem ec2-user@YOUR_EC2_IP
```

**View running containers:**
```bash
docker ps
```

**Expected output:**
```
CONTAINER ID   IMAGE                     STATUS        PORTS
abc123         username/fastapi-app      Up 2 hours    0.0.0.0:8000->8000/tcp
```

**View application logs:**
```bash
# View recent logs
docker logs fastapi-app

# Follow logs in real-time
docker logs -f fastapi-app

# View last 100 lines
docker logs --tail 100 fastapi-app
```

**Restart container if needed:**
```bash
docker restart fastapi-app
```

**Check disk space:**
```bash
df -h

# Clean up old Docker images on EC2
docker system prune -a
```

**Monitor resource usage:**
```bash
# CPU and memory usage
docker stats

# Container health
docker inspect fastapi-app | grep -A 10 Health
```

### **GitHub Actions Best Practices**

**1. Use Secrets for Sensitive Data**
- ✅ DO: Store tokens, passwords, SSH keys in secrets
- ❌ DON'T: Hardcode credentials in workflow files
- ❌ DON'T: Commit secrets to repository
- ✅ DO: Use environment variables from secrets

**2. Cache Dependencies for Faster Builds**
Already configured in our workflow:
```yaml
- name: Set up Python
  uses: actions/setup-python@v4
  with:
    python-version: '3.11'
    cache: 'pip'  # ← This caches pip dependencies
```

**Benefits:**
- First build: ~2 minutes (downloads all dependencies)
- Subsequent builds: ~30 seconds (uses cache)
- Saves GitHub Actions minutes

**3. Use workflow_dispatch for Manual Triggers**
Already configured:
```yaml
on:
  workflow_dispatch:  # ← Allows manual trigger
```

**Use cases:**
- Emergency hotfix deployment
- Testing workflow changes
- Force rebuild without code changes

**4. Test Locally Before Pushing**
Always run these locally first:
```bash
pytest app/
flake8 app/
docker build -t test .
```

**Saves:**
- GitHub Actions minutes
- Time waiting for pipeline
- Embarrassment from broken builds

**5. Use Meaningful Commit Messages**
**Good examples:**
```
feat: add user authentication endpoint
fix: resolve race condition in cache invalidation
docs: update API documentation for v2.0
refactor: extract database logic into separate module
```

**Bad examples:**
```
fixed stuff
changes
update
asdf
```

**Why it matters:**
- GitHub Actions logs show commit messages
- Easy to identify what broke the build
- Better team collaboration

**6. Monitor GitHub Actions Usage**

**Check your usage:**
- Go to GitHub → Settings → Billing → Plans and usage
- View: Actions minutes used / 2000 minutes available (free tier)

**Free tier limits:**
- 2,000 minutes/month for private repositories
- Unlimited for public repositories
- Minutes consumed faster on macOS/Windows runners (2x-10x multiplier)

**Tips to save minutes:**
- Use `ubuntu-latest` runner (cheapest, 1x multiplier)
- Cache dependencies aggressively
- Only run workflows when needed (use `paths-ignore`)
- Use matrix builds sparingly

**7. Set Up Branch Protection Rules**

**Configure on GitHub:**
- Go to Settings → Branches → Add rule
- Branch name pattern: `main`
- Enable:
  - ✅ Require status checks to pass before merging
  - ✅ Require branches to be up to date before merging
  - ✅ Require conversation resolution before merging
  - ✅ Include administrators

**Effect:**
- Cannot merge to main if tests fail
- Prevents accidental direct pushes to main
- Forces code review process

**8. Pin Action Versions**

**Good (specific version):**
```yaml
uses: actions/checkout@v4
uses: docker/build-push-action@v5
```

**Bad (floating latest):**
```yaml
uses: actions/checkout@latest  # Risky!
```

**Why:**
- Prevents breaking changes from action updates
- Reproducible builds
- Can test updates in PR before adopting

### **Common Issues and Solutions**

**Issue: "Error: Process completed with exit code 1"**
- **Cause**: Tests or linting failed
- **Solution**: Click on the failed step in Actions logs to see exact error

**Issue: Docker Hub rate limit exceeded**
```
Error: toomanyrequests: You have reached your pull rate limit
```
- **Cause**: Too many image pulls from Docker Hub
- **Solution**: Log in to Docker Hub locally or upgrade account

**Issue: Workflow runs on every push (too many builds)**
- **Cause**: No path filtering configured
- **Solution**: Add `paths-ignore` to workflow:
  ```yaml
  on:
    push:
      paths-ignore:
        - '**.md'
        - 'docs/**'
  ```

**Issue: "Resource not accessible by integration" error**
- **Cause**: Insufficient GitHub token permissions
- **Solution**: Grant necessary permissions in workflow:
  ```yaml
  permissions:
    contents: read
    packages: write
  ```

**Issue: Secrets not working in forked repositories**
- **Cause**: GitHub doesn't share secrets with forks for security
- **Solution**: Use pull_request_target instead of pull_request (with caution)

---

## **Full Pipeline Visualization**

```mermaid
graph TD
    A[Developer pushes code] --> B{GitHub Actions Triggered}
    
    B --> C[Test Job: ubuntu-latest]
    C --> C1[Checkout code]
    C --> C2[Setup Python 3.11]
    C --> C3[Install dependencies]
    C --> C4[Run flake8 linting]
    C --> C5[Run pytest with coverage]
    C --> C6[Upload coverage to Codecov]
    
    C5 -->|Tests pass| D[Build & Push Job]
    C5 -->|Tests fail| E[Pipeline stops]
    
    D --> D1[Set up Docker Buildx]
    D --> D2[Login to Docker Hub]
    D --> D3[Extract metadata]
    D --> D4[Build & push multi-platform image]
    
    D4 --> F[Deploy to EC2 Job]
    F --> F1[SSH to EC2]
    F --> F2[Pull latest image]
    F --> F3[Stop old container]
    F --> F4[Start new container]
    F --> F5[Verify deployment health]
    
    style A fill:#24292e,stroke:#24292e,stroke-width:2px,color:#fff
    style B fill:#0366d6,stroke:#0366d6,stroke-width:2px,color:#fff
    style C fill:#28a745,stroke:#28a745,stroke-width:2px,color:#fff
    style D fill:#f66a0a,stroke:#f66a0a,stroke-width:2px,color:#fff
    style F fill:#6f42c1,stroke:#6f42c1,stroke-width:2px,color:#fff
    style E fill:#dc3545,stroke:#dc3545,stroke-width:4px,color:#fff
```

---

## **Conclusion**

Congratulations on completing Lab 10! You've successfully built a **production-ready CI/CD pipeline** using GitHub Actions.

### **What You've Accomplished**

✅ **Automated Testing**: Every code change is automatically tested before deployment
✅ **Code Quality Enforcement**: Linting prevents style issues from reaching production
✅ **Continuous Integration**: Tests run on every push and pull request
✅ **Continuous Deployment**: Successful builds automatically deploy to Docker Hub
✅ **Infrastructure as Code**: Entire pipeline defined in version-controlled YAML
✅ **Zero Trust Deployment**: Tests must pass before any code reaches production

### **Key Metrics Your Pipeline Achieves**

- **Deployment Time**: Manual 30+ minutes → Automated 5-7 minutes (6x faster)
- **Deployment Frequency**: Once per week → Multiple times per day
- **Failure Rate**: High (untested code) → Low (all code tested before deployment)
- **Recovery Time**: Hours (manual rollback) → Minutes (automated rollback)
- **Confidence**: Low (hoping it works) → High (verified by tests)

### **Real-World Impact**

This is **not a toy project**. The CI/CD pipeline you've built uses the same patterns as:

- **Netflix**: Deploys 4000+ times per day with automated testing
- **Amazon**: Deploys every 11.6 seconds on average
- **Facebook**: 10,000+ commits per day with automated CI/CD
- **GitHub**: Every merged PR deploys automatically
- **Stripe**: Zero-downtime deployments with automated rollback

You've learned the **industry standard** for modern software development.

### **What Makes This Pipeline Production-Ready**

1. **Automated Testing**: Catches bugs before users see them
2. **Linting**: Maintains code quality and consistency
3. **Docker**: Consistent environments from dev to production
4. **Secrets Management**: Credentials never exposed in code
5. **Multi-stage Jobs**: Tests must pass before deployment
6. **Health Checks**: Verifies deployment succeeded
7. **Audit Trail**: Every deployment logged in GitHub
8. **Rollback Capability**: Can revert to any previous version

### **Skills You've Mastered**

- ✅ GitHub Actions workflow syntax
- ✅ CI/CD pipeline design
- ✅ Docker image building and publishing
- ✅ Automated testing with pytest
- ✅ Code linting with flake8
- ✅ Secrets management
- ✅ Multi-stage deployments
- ✅ Infrastructure automation

### **Next Steps to Level Up**

**1. Enhance Your Pipeline (Immediate)**
- Add performance testing (load testing)
- Implement security scanning (Trivy, Snyk)
- Add code coverage requirements (minimum 80%)
- Set up Codecov for coverage visualization

**2. Multi-Environment Setup (Intermediate)**
- Create staging environment
- Implement blue-green deployments
- Add manual approval gates for production
- Set up environment-specific configurations

**3. Advanced Practices (Long-term)**
- Implement feature flags (LaunchDarkly, Unleash)
- Add monitoring and observability (Prometheus, Grafana)
- Set up alerting (PagerDuty, Opsgenie)
- Implement canary deployments
- Add chaos engineering tests

**4. Team Collaboration (Professional)**
- Set up branch protection rules
- Require code reviews before merging
- Implement semantic versioning
- Add automated changelog generation
- Set up Slack/Discord notifications

### **Resources for Continued Learning**

**GitHub Actions:**
- Official docs: https://docs.github.com/en/actions
- Awesome Actions: https://github.com/sdras/awesome-actions
- Actions Marketplace: https://github.com/marketplace?type=actions

**CI/CD Best Practices:**
- "Continuous Delivery" by Jez Humble
- "The Phoenix Project" by Gene Kim
- DevOps Handbook

**Testing:**
- "Growing Object-Oriented Software, Guided by Tests"
- FastAPI Testing: https://fastapi.tiangolo.com/tutorial/testing/

### **Troubleshooting Resources**

If you encounter issues:

1. **Check GitHub Actions logs**: Click on failed workflow → failing step → read full error
2. **Test locally first**: Run same commands locally to reproduce issue
3. **Search GitHub Community**: https://github.community/
4. **Read action documentation**: Each action has docs with examples
5. **Check status pages**:
   - GitHub Status: https://www.githubstatus.com/
   - Docker Hub Status: https://status.docker.com/

### **Final Thoughts**

You've just learned one of the **most valuable skills** in software engineering: **automation**.

Every hour you spend setting up CI/CD saves hundreds of hours of manual testing and deployment. More importantly, it gives you the confidence to deploy code frequently without fear of breaking production.

This pipeline is the foundation. Build on it, customize it, and make it your own. The patterns you've learned here apply to **any** programming language, **any** framework, and **any** cloud provider.

**Welcome to the world of modern DevOps!** 🚀

---

## **Lab Summary Checklist**

Before moving to the next lab, ensure you've completed:

- [ ] Created FastAPI application with health checks
- [ ] Written tests for all endpoints
- [ ] Set up GitHub repository
- [ ] Configured GitHub secrets and variables
- [ ] Created GitHub Actions workflow
- [ ] Tested pipeline locally (linting + tests + Docker)
- [ ] Pushed code and watched pipeline run
- [ ] Verified Docker image on Docker Hub
- [ ] Tested at least 2 pipeline scenarios (success + failure)
- [ ] Understand how CI/CD prevents broken deployments

**If you've checked all boxes, you're ready for the next challenge!**

### **Share Your Success**

Built your first CI/CD pipeline? Share it!

- Tweet about it with #FastAPI #GitHub Actions #DevOps
- Add the project to your portfolio
- Write a blog post about your experience
- Help others in the community learn CI/CD

**You did it! 🎉**