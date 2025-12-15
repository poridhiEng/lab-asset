# Lab 04: Architectural Diagrams

## 1. Helm Repository Architecture

```mermaid
graph TB
    subgraph "GitHub Repository"
        GH[helm-charts repo]
        IDX[index.yaml]
        PKG1[welcome-app-1.0.0.tgz]
        PKG2[welcome-app-1.1.0.tgz]
        GH --> IDX
        GH --> PKG1
        GH --> PKG2
    end

    subgraph "GitHub Pages"
        GP[Static File Server]
        URL[https://user.github.io/helm-charts/]
    end

    subgraph "Helm Client"
        HC[helm CLI]
        CACHE[Local Repo Cache]
    end

    GH -->|Deploy| GP
    GP -->|Serves| URL
    HC -->|helm repo add| URL
    URL -->|index.yaml| CACHE
    HC -->|helm install| PKG1
    HC -->|helm install| PKG2
```

## 2. Package and Publish Workflow

```mermaid
flowchart LR
    subgraph "Development"
        A[Chart Source<br/>welcome-app/]
        B[helm lint]
        C[helm package]
        D[welcome-app-1.0.0.tgz]
    end

    subgraph "Repository"
        E[helm repo index]
        F[index.yaml]
        G[Git Push]
    end

    subgraph "GitHub Pages"
        H[Published Repository]
    end

    A --> B --> C --> D
    D --> E --> F
    D --> G
    F --> G --> H
```

## 3. Semantic Versioning

```mermaid
graph TD
    V1[v1.0.0<br/>Initial Release]

    V1 -->|Bug Fix| V101[v1.0.1<br/>Patch]
    V1 -->|New Feature| V110[v1.1.0<br/>Minor]
    V1 -->|Breaking Change| V200[v2.0.0<br/>Major]

    V101 -->|"Fixed typo in ConfigMap"| E1[Example]
    V110 -->|"Added NodePort support"| E2[Example]
    V200 -->|"Changed values.yaml structure"| E3[Example]

    style V1 fill:#4CAF50
    style V101 fill:#2196F3
    style V110 fill:#FF9800
    style V200 fill:#f44336
```

## 4. Index.yaml Structure

```mermaid
graph TB
    subgraph "index.yaml"
        ROOT[apiVersion: v1<br/>generated: timestamp]

        subgraph "entries"
            CHART[welcome-app]

            subgraph "versions"
                V1[version: 1.0.0<br/>appVersion: 1.0.0<br/>digest: sha256:abc...]
                V2[version: 1.1.0<br/>appVersion: 1.1.0<br/>digest: sha256:def...]
            end
        end
    end

    ROOT --> CHART
    CHART --> V1
    CHART --> V2

    V1 -->|urls| U1[welcome-app-1.0.0.tgz]
    V2 -->|urls| U2[welcome-app-1.1.0.tgz]
```

## 5. Client Repository Interaction

```mermaid
sequenceDiagram
    participant User
    participant Helm as Helm CLI
    participant Repo as GitHub Pages
    participant K8s as Kubernetes

    User->>Helm: helm repo add myrepo <url>
    Helm->>Repo: GET /index.yaml
    Repo-->>Helm: index.yaml content
    Helm->>Helm: Cache repository info

    User->>Helm: helm search repo myrepo
    Helm-->>User: List available charts

    User->>Helm: helm install app myrepo/welcome-app
    Helm->>Repo: GET /welcome-app-1.1.0.tgz
    Repo-->>Helm: Chart package
    Helm->>Helm: Extract and render templates
    Helm->>K8s: Apply manifests
    K8s-->>Helm: Resources created
    Helm-->>User: Installation complete
```

## 6. Repository Update Flow

```mermaid
flowchart TD
    subgraph "Version 1.0.0 Exists"
        A1[index.yaml with v1.0.0]
        P1[welcome-app-1.0.0.tgz]
    end

    subgraph "Create Version 1.1.0"
        B1[Update Chart.yaml<br/>version: 1.1.0]
        B2[helm package]
        B3[welcome-app-1.1.0.tgz]
    end

    subgraph "Merge Index"
        C1[helm repo index --merge]
        C2[Updated index.yaml<br/>v1.0.0 + v1.1.0]
    end

    subgraph "Publish"
        D1[git add & commit]
        D2[git push]
        D3[GitHub Pages Updated]
    end

    A1 --> C1
    B1 --> B2 --> B3 --> C1
    P1 --> D1
    B3 --> D1
    C1 --> C2 --> D1 --> D2 --> D3
```

## 7. Multi-Version Installation

```mermaid
graph LR
    subgraph "Repository"
        R[myrepo]
        R --> V1[v1.0.0]
        R --> V2[v1.1.0]
        R --> V3[v2.0.0]
    end

    subgraph "Installations"
        I1[helm install app-prod<br/>--version 1.0.0]
        I2[helm install app-staging<br/>--version 1.1.0]
        I3[helm install app-dev<br/>latest v2.0.0]
    end

    subgraph "Kubernetes"
        P1[app-prod<br/>v1.0.0]
        P2[app-staging<br/>v1.1.0]
        P3[app-dev<br/>v2.0.0]
    end

    V1 --> I1 --> P1
    V2 --> I2 --> P2
    V3 --> I3 --> P3
```

## 8. GitHub Pages Setup

```mermaid
flowchart TB
    subgraph "GitHub Repository Settings"
        A[Settings]
        B[Pages]
        C[Source: Deploy from branch]
        D[Branch: main]
        E[Folder: / root]
    end

    subgraph "Repository Contents"
        F[main branch]
        G[index.yaml]
        H[*.tgz files]
    end

    subgraph "Published Site"
        I[https://user.github.io/helm-charts/]
        J[/index.yaml]
        K[/welcome-app-1.0.0.tgz]
    end

    A --> B --> C --> D --> E
    F --> G
    F --> H
    E -->|Deploy| I
    G --> J
    H --> K
```

## 9. Complete Workflow Overview

```mermaid
flowchart TB
    subgraph "1. Development"
        DEV1[Create/Update Chart]
        DEV2[helm lint]
        DEV3[helm template]
    end

    subgraph "2. Package"
        PKG1[Update version in Chart.yaml]
        PKG2[helm package]
        PKG3[chart-x.y.z.tgz]
    end

    subgraph "3. Index"
        IDX1[helm repo index]
        IDX2[--merge for updates]
        IDX3[index.yaml]
    end

    subgraph "4. Publish"
        PUB1[git add]
        PUB2[git commit]
        PUB3[git push]
        PUB4[GitHub Pages]
    end

    subgraph "5. Consume"
        CON1[helm repo add]
        CON2[helm repo update]
        CON3[helm search repo]
        CON4[helm install]
    end

    DEV1 --> DEV2 --> DEV3 --> PKG1
    PKG1 --> PKG2 --> PKG3
    PKG3 --> IDX1
    IDX1 --> IDX2 --> IDX3
    PKG3 --> PUB1
    IDX3 --> PUB1 --> PUB2 --> PUB3 --> PUB4
    PUB4 --> CON1 --> CON2 --> CON3 --> CON4
```
