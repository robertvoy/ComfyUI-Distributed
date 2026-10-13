## Worker Setup Guide

**Master**: The main ComfyUI instance that coordinates and distributes work. This is where you load workflows, manage the queue, and view results.

**Worker**: A ComfyUI instance that receives and processes tasks from the master. Workers handle just the GPU computation and send results back to the master. You can have multiple workers connected to a single master, each utilizing their own GPU.

<img width="600" src="https://github.com/user-attachments/assets/609c42aa-8a1c-4a3f-939e-f3552fa1d54f" />

### Master participation modes

The master can either contribute GPU work or stay in **orchestrator-only** mode:

- **Participating**: Master renders alongside workers, useful when you want every available GPU.
- **Orchestrator-only**: Master sends jobs to selected workers but skips local rendering. Enable this by opening the Distributed panel and unchecking the master toggle. The master card will display *“Master disabled: running as orchestrator only.”*
- **Fallback**: If orchestrator-only is enabled but no workers remain selected, the master automatically re-enables execution to guarantee the workflow still runs. The UI shows a green *“Master fallback execution active”* badge so you know work is executing locally again.

### Types of Workers

- **Local workers**: Additional GPUs on the same machine as the master
- **Remote workers**: GPUs on different computers within your network
- **Cloud workers**: GPUs hosted on cloud services like Runpod

## Local workers

<img align="right" width="200" src="https://github.com/user-attachments/assets/651e4912-7c23-4e32-bd88-250f5175e129" />

> These are added automatically on first launch, but you can add them manually if you need to.


📺 [Watch Tutorial](https://youtu.be/p6eE3IlAbOs?si=K7Km0_flmPHwRQwz&t=43)

1. **Open** the Distributed GPU panel.
2. **Click** "Add Worker" in the UI.
3. **Configure** your local worker:
   - **Name**: A descriptive name for the worker (e.g., "Studio PC 1")
   - **Port**: A unique port number for this worker (e.g., 8189, 8190...).
   - **CUDA Device**: The GPU index from `nvidia-smi` (e.g., 0, 1).
   - **Extra Args**: Optional ComfyUI arguments for this specific worker.
4. **Save** and  launch the local worker.

## Remote workers

<img align="right" width="200"  src="https://github.com/user-attachments/assets/84291921-c44e-4556-94f2-a3b16500f4f9" />


> ComfyUI instances running on completely different computers on your network. These allow you to harness GPU power from other machines. Remote workers must be manually started on their respective computers and are connected via IP address.

📺 [Watch Tutorial](https://youtu.be/p6eE3IlAbOs?si=Oxj3EzPyf4jKDvfG&t=140)

**On the Remote Worker Machine:**
1. **Launch** ComfyUI with the `--listen --enable-cors-header` arguments. ⚠️ **Required!**
   - This ComfyUI instance will serve as a worker for your main master.
2. *Optionally* add additional local workers on this machine if it has multiple GPUs:
   - Access the Distributed GPU panel in this ComfyUI instance
   - Add workers for any additional GPUs (if they haven't been added automatically)
   - Make sure they have `--listen` set in `Extra Args`
   - Launch them
3. **Open** the ComfyUI port (e.g., 8188) and any additional worker ports (e.g., 8189, 8190) in the firewall.
  
**On the Main Machine:**
1. **Launch** ComfyUI with `--enable-cors-header` launch argument.
2. **Open** the Distributed GPU panel (sidebar on the left).
3. **Click** "Add Worker."
4. **Choose** "Remote".
5. **Configure** your remote worker:
   - **Name**: A descriptive name for the worker (e.g., "Server Rack GPU 0")
   - **Host**: The remote worker's IP address.
   - **Port**: The port number used when launching ComfyUI on the remote master/worker (e.g., 8188).
6. **Save** the remote worker configuration.
  
## Cloud workers

<img align="right" width="200"  src="https://github.com/user-attachments/assets/a053f3ae-22f0-4e1c-8f2e-f26a1f660adf" />

> ComfyUI instances running on a cloud service like Runpod. 

For faster result transfers and lower bandwidth usage, see [Compressed Result Transfer](compressed-result-transfer.md).

### Deploy Cloud Worker on Runpod

📺 [Watch Tutorial](https://www.youtube.com/watch?v=wxKKWMQhYTk)

**On Runpod:**
> If using your own template, launch ComfyUI with `--listen --enable-cors-header` and clone `ComfyUI-Distributed` into `custom_nodes`. ⚠️ **Required!**

1. Register a [Runpod](https://get.runpod.io/0bw29uf3ug0p) account.
2. On Runpod, go to Storage > New Network Volume and create a volume that will store the models you need. Start with 40 GB, you can always add more later. Learn more [about Network Volumes](https://docs.runpod.io/pods/storage/create-network-volumes).
3. Use the [ComfyUI Distributed Pod](https://console.runpod.io/deploy?template=m21ynvo8yo&ref=0bw29uf3ug0p) template.
4. Make sure your Network Volume is mounted and choose a suitable GPU.
> ⚠️ To use the ComfyUI Distributed Pod template, you will need to filter instances by CUDA 12.8 (add filter in Additional Filters).
6. Press Edit Template to configure the pod's Environment Variables:
	- CIVITAI_API_TOKEN: [get your token here](https://civitai.com/user/account)
	- HF_API_TOKEN: [get your token here](https://huggingface.co/settings/tokens)
	- SAGE_ATTENTION: optional optimisation (set to true/false)
5. Deploy your pod.
6. Connect to your pod using JupyterLabs. This gives us access to the pod's file system.
7. Download models into `/workspace/ComfyUI/models/` (these will remain on your network drive even after you terminate the pod). Example commands below:
```
# Download from CivitAI
comfy model download --url https://civitai.com/api/download/models/1759168 --relative-path /workspace/ComfyUI/models/checkpoints --set-civitai-api-token $CIVITAI_API_TOKEN
# Download model from Hugging Face
comfy model download --url https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors --relative-path /workspace/ComfyUI/models/unet --set-hf-api-token $HF_API_TOKEN
```
> ℹ️ Use [this guide](model-download-script.md) to make this process easy. It will generate a shell script that automatically downloads the models for a given workflow.
9. Access ComfyUI through the Runpod URL.
10. Download any additional custom nodes you need using the ComfyUI Manager.

**On the Main Machine:**
1. **Launch** a Cloudflare tunnel.
   - Download from here: [https://github.com/cloudflare/cloudflared/releases](https://github.com/cloudflare/cloudflared/releases)
	- Then run, for example: `cloudflared-windows-amd64.exe tunnel --url http://localhost:8188`
> ℹ️ Cloudflare tunnels create secure connections without exposing ports directly to the internet and are required for Cloud Workers.
2. **Copy** the Cloudflare address
3. **Launch** ComfyUI with `--enable-cors-header` launch argument.
4. **Open** the Distributed GPU panel (sidebar on the left).
5. **Edit** the Master's settings to change the host address to the Cloudflare address.
6. **Click** "Add Worker."
7. **Choose** "Cloud".
8. **Configure** your cloud worker:
	- **Host**: The ComfyUI Runpod address. For example: `wcegfo9tbbml9l-8188.proxy.runpod.net`
	- **Port**: 443
9. **Save** the remote worker configuration.

---

### Deploy Cloud Worker on Other Platforms

**On the Cloud Worker machine:**
   - Your cloud worker container needs to have the same models and custom nodes as the workflow you want to run on your local machine.
   - If your cloud platform doesn't provide a secure connection, use Cloudflare to create a tunnel for the worker. Each GPU needs their own tunnel for their respective port.
	- For example: `./cloudflared tunnel --url http://localhost:8188`
1. **Launch** ComfyUI with the `--listen --enable-cors-header` arguments. ⚠️ **Required!**
2. **Add** workers in the UI panel if the cloud machine has more than one GPU.
   - Make sure that they also have `--listen` set in `Extra Args`.
   - Then launch them.
  
**On the Main Machine:**
1. **Launch** a Cloudflare tunnel on your local machine.
   - Download from here: [https://github.com/cloudflare/cloudflared/releases](https://github.com/cloudflare/cloudflared/releases)
   - Then run, for example: `cloudflared-windows-amd64.exe tunnel --url http://localhost:8188`
2. **Copy** the Cloudflare address
3. **Launch** ComfyUI with `--enable-cors-header` launch argument.
4. **Open** the Distributed GPU panel (sidebar on the left).
5. **Edit** the Master's host address and replace it with the Cloudflare address.
6. **Click** "Add Worker."
7. **Choose** "Cloud".
8. **Configure** your cloud worker:
   - **Host**: The remote worker's IP address/domain
   - **Port**: 443
9. **Save** the remote worker configuration.
