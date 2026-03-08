# Jenkins on Kubernetes Setup Guide

**Environment:** Vanilla kubeadm (Kubespray) | Separate control plane + worker node | Helm installed
**Worker Node:** `minisforum-gray-1` | IP: `10.0.0.169` | 16 CPU | 32GB RAM | Ubuntu 24.04.3 LTS | k8s v1.33.7

---

## Prerequisites

Before deploying Jenkins, verify your worker node is healthy and has sufficient resources.

```bash
kubectl get nodes
kubectl describe node minisforum-gray-1
```

Your node (`minisforum-gray-1` at `10.0.0.169`) is already confirmed healthy:
- ✅ Status: `Ready`
- ✅ CPU: 16 cores (only 3% in use)
- ✅ Memory: ~32GB RAM (only 1% in use)
- ✅ Disk: ~93GB available
- ✅ No taints — pods will schedule without any extra configuration

---

## Step 1: Install a Storage Provisioner

Vanilla kubeadm ships with no default storage class. Without one, Jenkins' PersistentVolumeClaim (PVC) will stay stuck in `Pending` and the pod will never start.

The easiest solution for a single worker node is Rancher's **local-path-provisioner**, which automatically provisions storage from the node's local disk.

**Install the provisioner:**
```bash
kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/master/deploy/local-path-storage.yaml
```

**Verify it's running:**
```bash
kubectl get pods -n local-path-storage
```

**Set it as the default storage class:**
```bash
kubectl patch storageclass local-path -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
```

**Confirm the default is set (look for the `(default)` label):**
```bash
kubectl get storageclass
```

> ⚠️ **Note:** `local-path` stores data on the worker node's local disk. This means data is tied to that specific node. For a single-node setup this is fine, but it is not suitable for multi-node clusters where pods can be rescheduled to different nodes.

---

## Step 2: Install NGINX Ingress Controller

The ingress controller is what routes external HTTP traffic into your cluster and exposes the Jenkins UI on your local network.

**Add the Helm repo:**
```bash
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update
```

**Install the ingress controller:**
```bash
helm install ingress-nginx ingress-nginx/ingress-nginx \
  -n ingress-nginx \
  --create-namespace
```

**Verify the controller pod is running:**
```bash
kubectl get pods -n ingress-nginx
```

**Check that an external IP has been assigned (this will be your worker node's IP):**
```bash
kubectl get svc -n ingress-nginx
```

> ℹ️ On a bare-metal kubeadm cluster, the `EXTERNAL-IP` for the ingress service may show as `<pending>`. This is normal without a cloud load balancer. Jenkins will still be accessible directly via your worker node's IP on port 80.

---

## Step 3: Install Jenkins via Helm

**Add the Jenkins Helm repo:**
```bash
helm repo add jenkins https://charts.jenkins.io
helm repo update
```

**Create a dedicated namespace:**
```bash
kubectl create namespace jenkins
```

**Install Jenkins:**
```bash
helm install jenkins jenkins/jenkins -n jenkins
```

The Helm chart will automatically:
- Deploy the Jenkins controller pod
- Create a PersistentVolumeClaim using the default storage class from Step 1
- Configure the Kubernetes plugin for dynamic agent pods
- Set up a ClusterIP service for the Jenkins UI

**Watch the pod come up:**
```bash
kubectl get pods -n jenkins -w
```

Wait until the pod shows `Running` and `1/1` before proceeding. This can take a couple of minutes on first start as Jenkins initialises.

---

## Step 4: Retrieve the Admin Password

The initial admin password is stored as a Kubernetes secret. Run the following to retrieve it:

```bash
kubectl exec -n jenkins -it svc/jenkins -c jenkins -- \
  /bin/cat /run/secrets/additional/chart-admin-password
```

Copy this password — you'll need it to log in for the first time.

> The default admin username is `admin`.

---

## Step 5: Access the Jenkins UI

### Option A: Port Forward (Quick Test)

The fastest way to verify Jenkins is working without any ingress configuration:

```bash
kubectl port-forward -n jenkins svc/jenkins 8080:8080
```

Then open your browser and navigate to:
```
http://localhost:8080
```

> This only works from the machine running the `kubectl` command. Stop it with `Ctrl+C`.

### Option B: Ingress (Persistent Access on Your Network)

To make Jenkins permanently accessible on your local network, create an ingress resource.

Create a file called `jenkins-ingress.yaml`:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: jenkins
  namespace: jenkins
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "0"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "600"
spec:
  ingressClassName: nginx
  rules:
    - host: jenkins.local       # Or replace with 10.0.0.169 to access via worker node IP directly
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: jenkins
                port:
                  number: 8080
```

Apply it:
```bash
kubectl apply -f jenkins-ingress.yaml
```

Then either:
- Add `jenkins.local` pointing to `10.0.0.169` in your local `/etc/hosts` file, **or**
- Replace `host: jenkins.local` with no host rule to access via the worker node IP directly

---

## Step 6: Verify the Kubernetes Plugin

The Jenkins Helm chart pre-installs the **Kubernetes plugin**, which allows Jenkins to dynamically spin up agent pods for builds and tear them down when done.

To confirm it's configured:

1. Log into Jenkins
2. Navigate to **Manage Jenkins → Manage Nodes and Clouds → Configure Clouds**
3. You should see a Kubernetes cloud entry already configured pointing to the in-cluster API

> ℹ️ By default, agent pods are created in the `jenkins` namespace. The Helm chart also sets up the necessary RBAC (service account + role bindings) for this automatically.

---

## Useful Commands Reference

| Task | Command |
|---|---|
| Check all Jenkins resources | `kubectl get all -n jenkins` |
| View Jenkins pod logs | `kubectl logs -n jenkins -l app.kubernetes.io/name=jenkins` |
| Restart Jenkins pod | `kubectl rollout restart deployment -n jenkins jenkins` |
| Get admin password again | `kubectl exec -n jenkins -it svc/jenkins -c jenkins -- /bin/cat /run/secrets/additional/chart-admin-password` |
| Uninstall Jenkins | `helm uninstall jenkins -n jenkins` |
| Check PVC status | `kubectl get pvc -n jenkins` |

---

## Troubleshooting

**Pod stuck in `Pending`**
```bash
kubectl describe pod -n jenkins <pod-name>
```
Usually caused by the PVC not binding. Check that the storage class is set as default (`kubectl get storageclass`).

**Pod stuck in `CrashLoopBackOff`**
```bash
kubectl logs -n jenkins <pod-name> -c jenkins
```
Often a permissions issue on the persistent volume. Check the events in `kubectl describe pod`.

**Can't reach Jenkins UI**
- Confirm the pod is `Running`: `kubectl get pods -n jenkins`
- Confirm the service exists: `kubectl get svc -n jenkins`
- Try port-forwarding as a baseline test (Step 5, Option A)

---

*Generated: March 2026*
