# Ingress Annotations & Advanced Configuration

The Kubernetes Ingress specification defines basic routing—path matching, host matching, and TLS termination. But real-world applications need more: URL rewriting, rate limiting, timeouts, and custom headers. These features aren't part of the Ingress spec because different Ingress Controllers implement them differently.

![](./images/3.svg)

**Annotations** bridge this gap. They're key-value pairs in the Ingress metadata that configure controller-specific behavior. In this lab, you'll learn the most important NGINX Ingress Controller annotations for production deployments.

## Why Annotations Matter

Consider a common microservices scenario:

```
Client Request:  GET /api/users/123
Backend Expects: GET /users/123
```

Your API gateway exposes services under `/api/*`, but backend services don't know about this prefix—they expect requests at `/`. Without URL rewriting, the backend receives `/api/users/123` and returns 404.

The Ingress spec has no rewrite feature. NGINX Ingress Controller adds it through the `rewrite-target` annotation:

```yaml
annotations:
  nginx.ingress.kubernetes.io/rewrite-target: /$2
```

This pattern applies to rate limiting, timeouts, headers, and many other features—all configured through annotations.

## Common NGINX Annotations

| Annotation | Purpose |
|------------|---------|
| `rewrite-target` | Modify URL path before forwarding |
| `limit-rps` | Rate limit requests per second |
| `limit-connections` | Limit concurrent connections |
| `proxy-read-timeout` | Backend response timeout |
| `proxy-connect-timeout` | Backend connection timeout |
| `configuration-snippet` | Inject custom NGINX config |

Full list: [NGINX Ingress Annotations Documentation](https://kubernetes.github.io/ingress-nginx/user-guide/nginx-configuration/annotations/)


## Step 1: Install NGINX Ingress Controller

Install the NGINX Ingress Controller if not already installed:

```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.12.0/deploy/static/provider/cloud/deploy.yaml
```

Wait for the controller to be ready:

```bash
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=120s
```

## Step 2: Deploy Backend Applications

Create `apps.yaml`:

```yaml
# API Application - returns request path info
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-app
spec:
  replicas: 1
  selector:
    matchLabels:
      app: api
  template:
    metadata:
      labels:
        app: api
    spec:
      containers:
      - name: api
        image: hashicorp/http-echo
        args:
        - "-text={\"service\": \"api\", \"message\": \"Request received at API backend\"}"
        - "-listen=:8080"
        ports:
        - containerPort: 8080
---
apiVersion: v1
kind: Service
metadata:
  name: api-svc
spec:
  selector:
    app: api
  ports:
  - port: 80
    targetPort: 8080
---
# Echo Application - echoes headers and request info
apiVersion: apps/v1
kind: Deployment
metadata:
  name: echo-app
spec:
  replicas: 2
  selector:
    matchLabels:
      app: echo
  template:
    metadata:
      labels:
        app: echo
    spec:
      containers:
      - name: echo
        image: ealen/echo-server
        ports:
        - containerPort: 80
        env:
        - name: PORT
          value: "80"
---
apiVersion: v1
kind: Service
metadata:
  name: echo-svc
spec:
  selector:
    app: echo
  ports:
  - port: 80
    targetPort: 80
```

**Understanding the apps:**

- **api-app**: Simple HTTP echo service for testing rewrite
- **echo-app**: Full echo server that shows request headers, path, and more (useful for debugging)

Apply:

```bash
kubectl apply -f apps.yaml
```

Verify:

```bash
kubectl get pods
```

Start port-forward for testing:

```bash
kubectl port-forward -n ingress-nginx svc/ingress-nginx-controller 8080:80 &
```

## Step 3: URL Rewriting with rewrite-target

![](./images/1.svg)

The most important annotation. Without it, path-based routing breaks most backend services.

Create `rewrite-ingress.yaml`:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rewrite-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /$2
spec:
  ingressClassName: nginx
  rules:
  - http:
      paths:
      - path: /api(/|$)(.*)
        pathType: ImplementationSpecific
        backend:
          service:
            name: api-svc
            port:
              number: 80
      - path: /echo(/|$)(.*)
        pathType: ImplementationSpecific
        backend:
          service:
            name: echo-svc
            port:
              number: 80
```

**Understanding the rewrite:**

The path `/api(/|$)(.*)` uses regex capture groups:
- `(/|$)` - Matches `/` or end of string (capture group 1)
- `(.*)` - Matches everything after (capture group 2)

The `rewrite-target: /$2` sends only capture group 2 to the backend:

| Client Request | Backend Receives |
|----------------|------------------|
| `/api` | `/` |
| `/api/` | `/` |
| `/api/users` | `/users` |
| `/api/users/123` | `/users/123` |

Apply and test:

```bash
kubectl apply -f rewrite-ingress.yaml
```

```bash
# Test rewrite - /api/users becomes /users at backend
echo "=== Request to /api ==="
curl -s http://localhost:8080/api

echo -e "\n=== Request to /echo/test ==="
curl -s http://localhost:8080/echo/test | jq | head -20
```

The echo server shows the actual path received by the backend—it should be `/test`, not `/echo/test`.

## Step 4: Rate Limiting

![](./images/2.svg)

Protect your backend from too many requests. When the limit is exceeded, NGINX returns 503.

Create `ratelimit-ingress.yaml`:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ratelimit-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /$2
    nginx.ingress.kubernetes.io/limit-rps: "2"
    nginx.ingress.kubernetes.io/limit-connections: "5"
spec:
  ingressClassName: nginx
  rules:
  - http:
      paths:
      - path: /limited(/|$)(.*)
        pathType: ImplementationSpecific
        backend:
          service:
            name: echo-svc
            port:
              number: 80
```

**Understanding rate limiting:**

- `limit-rps: "2"` - Maximum 2 requests per second per client IP
- `limit-connections: "5"` - Maximum 5 concurrent connections per client IP

Apply and test:

```bash
kubectl delete ingress rewrite-ingress
kubectl apply -f ratelimit-ingress.yaml
```

```bash
# Send 10 rapid requests - some should fail with 503
echo "=== Rapid requests (expect some 503s) ==="
for i in {1..20}; do
  code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/limited/)
  echo "Request $i: HTTP $code"
done
```

You should see `HTTP 200` for the first few requests, then `HTTP 503` as the rate limit kicks in.

## Step 5: Timeouts

Configure how long NGINX waits for backend responses. This is critical for slow APIs or file uploads.

Create `timeout-ingress.yaml`:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: timeout-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /$2
    # Timeouts
    nginx.ingress.kubernetes.io/proxy-connect-timeout: "10"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "30"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "30"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
spec:
  ingressClassName: nginx
  rules:
  - http:
      paths:
      - path: /echo(/|$)(.*)
        pathType: ImplementationSpecific
        backend:
          service:
            name: echo-svc
            port:
              number: 80
```

**Understanding the annotations:**

- `proxy-connect-timeout: "10"` - Wait max 10 seconds to establish connection to backend
- `proxy-read-timeout: "30"` - Wait max 30 seconds for backend response
- `proxy-send-timeout: "30"` - Wait max 30 seconds to send request to backend
- `proxy-body-size: "10m"` - Allow request bodies up to 10 megabytes

Apply and test:

```bash
kubectl delete ingress ratelimit-ingress
kubectl apply -f timeout-ingress.yaml
```

```bash
# Test the endpoint
echo "=== Echo endpoint ==="
curl -s http://localhost:8080/echo/ | jq 
```

The echo server shows request details including headers that NGINX automatically forwards (like `X-Forwarded-For`, `X-Real-IP`).

## Step 6: Combined Configuration

In production, you typically combine multiple annotations. Here's a complete example:

Create `production-ingress.yaml`:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: production-ingress
  annotations:
    # URL Rewriting
    nginx.ingress.kubernetes.io/rewrite-target: /$2
    # Rate Limiting
    nginx.ingress.kubernetes.io/limit-rps: "2"
    nginx.ingress.kubernetes.io/limit-connections: "2"
    # Timeouts
    nginx.ingress.kubernetes.io/proxy-connect-timeout: "5"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "60"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "60"
    # Request body size
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
spec:
  ingressClassName: nginx
  rules:
  - http:
      paths:
      - path: /api(/|$)(.*)
        pathType: ImplementationSpecific
        backend:
          service:
            name: api-svc
            port:
              number: 80
      - path: /echo(/|$)(.*)
        pathType: ImplementationSpecific
        backend:
          service:
            name: echo-svc
            port:
              number: 80
```

Apply and test:

```bash
kubectl delete ingress timeout-ingress
kubectl apply -f production-ingress.yaml
```

```bash
# Test API with rewrite
echo "=== /api endpoint ==="
curl -s http://localhost:8080/api/

# Test echo - see forwarded headers
echo -e "\n=== /echo endpoint ==="
curl -s http://localhost:8080/echo/ | jq '.request.headers' | head -15

# Test rate limiting
echo -e "\n=== Rate limit test ==="
for i in {1..15}; do
  code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/api/)
  echo "Request $i: HTTP $code"
done
```

## Cleanup

```bash
kubectl delete ingress --all
kubectl delete -f apps.yaml
pkill -f "port-forward.*ingress-nginx"
```

## Conclusion

Annotations extend Ingress capabilities beyond the basic spec. The most essential NGINX Ingress annotations are:

- **rewrite-target** - Strip path prefixes for microservices routing
- **limit-rps / limit-connections** - Protect backends from request floods
- **proxy-read-timeout** - Handle slow backend responses
- **proxy-body-size** - Allow large file uploads

NGINX Ingress automatically forwards useful headers to backends (`X-Real-IP`, `X-Forwarded-For`, `X-Forwarded-Proto`), so you don't need custom configuration for basic header forwarding.

Remember that annotations are controller-specific. If you switch from NGINX to Traefik or another Ingress Controller, you'll need to update your annotations to match that controller's format.
