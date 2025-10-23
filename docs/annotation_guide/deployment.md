# Deployment Options for Annotation Platform

Choose the deployment method that best fits your data security requirements and infrastructure.

## Run Locally
Run the annotation platform on your local machine.

```python
from meta_evaluator import MetaEvaluator

evaluator = MetaEvaluator(project_dir="my_project", load=True)
evaluator.launch_annotator(port=8501)  # Access at http://localhost:8501
```

## Remote Access with ngrok
Share your annotation interface with remote annotators via public URL.

If you wish to share your annotation interface with remote annotators and your data is not classified, you can use ngrok to create a public URL that anyone with the link can access.

1. **Install ngrok:** [https://ngrok.com/download/](https://ngrok.com/download/)

2. **Authenticate:**
    ```bash
    ngrok authtoken YOUR_TOKEN  # Free account required
    ```

3. **Launch with ngrok:**
    ```python
    from meta_evaluator import MetaEvaluator

    evaluator = MetaEvaluator(project_dir="my_project", load=True)

    # Basic ngrok tunnel - creates a public URL
    evaluator.launch_annotator(
        port=8501,
        use_ngrok=True
    )
    ```

**How it works:**

ngrok creates a tunnel from a public URL (e.g., `https://abc123.ngrok.io`) to your local machine. Remote annotators can access the interface through this URL without needing VPN or complex network setup.

!!! warning "Data Security"
    **Only use ngrok if your data is not classified.** ngrok creates a publicly accessible URL that exposes your local annotation interface to the internet. Anyone with the URL can access your interface. 


**Advanced: Traffic Policy Files**

```python
evaluator.launch_annotator(
    port=8501,
    use_ngrok=True,
    traffic_policy_file="ngrok_policy.yaml"
)
```

**Traffic policy files** allow advanced configuration including login authentication, IP restrictions, custom headers, and more.
For examples and detailed configuration, see: [ngrok Traffic Policy Documentation](https://ngrok.com/docs/traffic-policy/)


## Docker Deployment

Deploy the annotation platform on any hosting provider (cloud servers, on-premise infrastructure, etc.).

Docker deployment offers several advantages, especially when working with sensitive or classified data. It allows you to deploy the annotation platform entirely within your own infrastructure and gate your data and annotations. 

Install Docker on your host environment (your local machine or server where Docker will run):

- **Docker**: [Install Docker](https://docs.docker.com/get-docker/)
- **Docker Compose**: Included with Docker Desktop, or [install separately](https://docs.docker.com/compose/install/)

**Download Docker templates:**

Download Dockerfile and docker-compose.yml templates:
```bash
curl -O https://raw.githubusercontent.com/govtech-responsibleai/meta-evaluator/refs/heads/main/docker/Dockerfile
curl -O https://raw.githubusercontent.com/govtech-responsibleai/meta-evaluator/refs/heads/main/docker/docker-compose.yml
```

Or manually download the files:

- Download manually: [Dockerfile](https://raw.githubusercontent.com/govtech-responsibleai/meta-evaluator/main/docker/Dockerfile)
- View template: [docker-compose.yml](https://github.com/govtech-responsibleai/meta-evaluator/blob/main/docker/docker-compose.yml)

**About the templates:**

- The **Dockerfile** template can be used directly as it downloads the necessary Python version and packages, but you may modify it according to your environment needs.
- The **docker-compose.yml** template must be modified as it contains project-specific configurations such as file paths, build context, and the command to run your annotation script.

!!! note
    Using Docker Compose is optional. Depending on the complexity of your setup, you may choose to use `docker build` and `docker run` commands directly instead.

### Local Machine

Ensure you have downloaded the Docker templates above and are in the directory containing the Dockerfile and docker-compose.yml.

1.  **Prepare your script:**

    Create your `run_annotation.py` script to load your task/data and launch the annotator. See the [Annotation Guide](annotation.md) on how to set up the script.

2.  **Build and run:**

    ```bash
    docker compose build
    docker compose up

    ```

3. **Access at `http://localhost:8501`**

### Server

1.  **SSH into your server:**

    ```bash
    ssh user@server-ip
    ```

2.  **Set up workspace on server:**

    Download the Docker templates (see instructions above) and upload them to your server. Ensure you are in the directory containing the Dockerfile and docker-compose.yml.

    ```bash
    # Create workspace directory
    mkdir workspace && cd workspace

    # Upload Dockerfile and docker-compose.yml to this directory
    ```

3.  **Prepare and upload your script and data:**

    Create your `run_annotation.py` script to load your task/data and launch the annotator. See the [Annotation Guide](annotation.md) on how to set up the script.

    ```bash
    scp run_annotation.py data.csv user@server-ip:~/workspace/
    ```

    Your project directory should look like this:

    ```
    ~/workspace/
    ├── Dockerfile             # Downloaded in step 2
    ├── docker-compose.yml     # Downloaded and configured in step 2
    ├── run_annotation.py      # Uploaded in step 3
    └── data.csv               # Uploaded in step 3
    ```

4.  **Build and Run:**
    ```bash
    # Build the image
    docker compose build

    # Run the annotation platform
    docker compose up
    ```

5.  **Access at `http://server-ip:8501`**

6.  **After annotation, copy project directory back to local:**
   ```bash
   # From your laptop - copy the project folder with annotations
   scp -r user@server-ip:~/workspace/my_project ./
   ```

Configure network access, authentication, and security based on your hosting provider's capabilities.

