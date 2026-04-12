Since this is your first time, let’s go through the absolute **zero-to-benchmark** guide for setting up your Google Cloud (GCloud) environment. 🕵🏽‍♂️ 

### **Step 1: Set Up Your Account & Project**
1.  **The Console**: Go to [console.cloud.google.com](https://console.cloud.google.com).
2.  **The Project**: At the top left, click the project dropdown and select **New Project**. Name it `maxtext-dbs-test`. 
3.  **Billing**: You **must** attach a billing account (Credit Card or Free Trial) to the project. Without this, you cannot use any GPUs or TPUs. 💰

### **Step 2: Enable the APIs (The "Engine Room")**
Google Cloud services are "off" by default. You need to turn them on:
1.  In the search bar at the top, type **Compute Engine API** and click **Enable**. 
2.  Also search for and enable the **Cloud TPU API**. 

### **Step 3: The Biggest Block: Requesting Quota**
New accounts have a "0" limit for GPUs and TPUs. You must ask Google to let you pay for them:
1.  Search for **Quotas & System Limits** in the top bar.
2.  **Filter by Metric**: 
    *   Search for `L4` (if you want the cheap GPU).
    *   Search for `v5e` (if you want the TPU).
3.  **Location**: Filter for `us-central1` or `us-east5`.
4.  **Action**: Check the box for your metric (e.g., `L4 GPUs`), click **Edit Quotas**, and request **1** (for L4) or **8** (for TPU). 🛡️ *Note: This can take 1–24 hours to be approved.*

### **Step 4: Create Your "Home" (The Cloud Storage Bucket)**
You need a place to save your results and the model checkpoints:
1.  Search for **Cloud Storage** in the console. 
2.  Click **Create Bucket**. Name it something unique (e.g., `maxtext-output-yourname`). Keep all other settings as default.

### **Step 5: Launch Your VM (The Easy Way)**
Wait until your quota (Step 3) is approved, then open the **Cloud Shell** (the `>_` terminal icon at the top right of the console window) and run:

**For a single L4 GPU (Cheapest/Easiest):**
```bash
gcloud compute instances create maxtext-vm \
    --zone=us-central1-a \
    --machine-type=g2-standard-4 \
    --image-family=common-cu121-debian-11-py310 \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE
```

### **Step 6: Connect & Install**
1.  **SSH**: In your Cloud Shell, run: `gcloud compute ssh maxtext-vm --zone=us-central1-a`. 
2.  **Git & MaxText**:
    ```bash
    git clone https://github.com/google/maxtext.git
    cd maxtext
    bash setup.sh  # This installs JAX, CUDA, and everything else!
    ```

### **Step 7: Run Your Benchmark!** 🏁
You are now ready to run the `decode.py` command we practiced earlier!
