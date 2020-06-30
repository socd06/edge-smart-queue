# edge-smart-queue

[![Python Version](https://img.shields.io/badge/Python-3.5|3.6-blue.svg)](https://shields.io/)
[![GitHub license](https://img.shields.io/github/license/socd06/edge-smart-queue)](https://github.com/socd06/edge-smart-queue/blob/master/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

The edge smart queue is an example business AI system. The app detects people in a frame and recommends them move to the next queue if the maximum determined by the business is reached. Read the proyect [WRITEUP](https://github.com/socd06/edge-smart-queue/blob/master/docs/WRITEUP-choose-the-right-hardware.pdf).

The [job submission script](https://github.com/socd06/edge-smart-queue/blob/master/scripts/queue_job.sh) uses the Intel DevCloud to run the [smart queue script](https://github.com/socd06/edge-smart-queue/blob/master/scripts/person_detect.py) using Intel OpenVINO.

## Requirements
Optional if using the Udacity classroom workspace. Otherwise, see the next subsections.

### Hardware
- Optional if ran with Intel DevCloud. DevCloud can be used to test on a variety of CPU, GPU, FPGA and VPU.
- Intel CPU, GPU, FPGA or VPU hardware required to run at the edge.

### Software
* [Intel® Distribution of OpenVINO™ toolkit 2020.2.120 release](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/choose-download.html)
* Python > 3.5, 3.6

## Run the application at the Edge

### Download model
Download the [person-detection-retail-0013](https://docs.openvinotoolkit.org/latest/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html) model

Go into installed OpenVINO folder

`cd /opt/intel/openvino/deployment_tools/tools/model_downloader`

Download the model using the model downloader script

`sudo ./downloader.py --name person-detection-retail-0013 -o ~/git/openvino-models`

**Note:** ***git*** is the folder name where I store my GitHub projects locally, you may use whichever folder you'd like.

This will download all available versions of the model into new intel/<model version> folders

### Neural Compute Stick 2

Go into repository folder, for example:

`cd ~/git/edge-smart-queue`

Then run the following to run the script on the manufacturing scenario:

`python scripts/person_detect.py --model ~/git/openvino-models/intel/person-detection-retail-0013/FP16/person-detection-retail-0013 --device MYRIAD --video original_videos/Manufacturing.mp4 --output_path results/ --max_people 6 --v_queue 0`

## scripts
### Person Detection
The [person_detect.py](https://github.com/socd06/edge-smart-queue/blob/master/scripts/person_detect.py) is the people counter (AKA Smart Queue app) file.

### Job Submission
The [queue_job.sh](https://github.com/socd06/edge-smart-queue/blob/master/scripts/queue_job.sh) submits the inference job to DevCloud.

## notebooks
* [Retail Scenario](https://github.com/socd06/edge-smart-queue/blob/master/notebooks/Retail_Scenario.ipynb)
* [Manufacturing Scenario](https://github.com/socd06/edge-smart-queue/blob/master/notebooks/Manufacturing_Scenario.ipynb)
* [Transportation Scenario](https://github.com/socd06/edge-smart-queue/blob/master/notebooks/Transportation_Scenario.ipynb)

## Run the application with Intel DevCloud
The figure below illustrates the user workflow for code development, job submission and viewing results.

![](https://github.com/socd06/edge-smart-queue/blob/master/images/How-DevCloud-works.svg)

<details open>
<summary>Credits</summary>
<br>
<a href="https://devcloud.intel.com/edge/">Intel DevCloud website</a>
</details>

### Submit to an Edge Compute Node with an Intel CPU

We write a script to submit a job to an
[IEI Tank* 870-Q170](https://software.intel.com/en-us/iot/hardware/iei-tank-dev-kit-core) edge node with an [Intel Core™ i5-6500TE processor](https://ark.intel.com/products/88186/Intel-Core-i5-6500TE-Processor-6M-Cache-up-to-3-30-GHz-).
The inference workload should run on the CPU.

```sh
cpu_job_id = !qsub queue_job.sh -d . -l nodes=1:tank-870:i5-6500te -F "[model_path] CPU [original_video_path] /data/queue_param/manufacturing.npy [output_path] 2" -N store_core
```

### Submit to an Edge Compute Node with CPU and IGPU

We write a script to submit a job to an [IEI Tank* 870-Q170](https://software.intel.com/en-us/iot/hardware/iei-tank-dev-kit-core) edge
node with an [Intel® Core i5-6500TE](https://ark.intel.com/products/88186/Intel-Core-i5-6500TE-Processor-6M-Cache-up-to-3-30-GHz-). The
inference workload should run on the Intel® HD Graphics 530 integrated GPU.

```sh
gpu_job_id = !qsub queue_job.sh -d . -l nodes=tank-870:i5-6500te:intel-hd-530 -F "[model_path] GPU [original_video_path] /data/queue_param/manufacturing.npy [output_path] 2" -N store_core
```

### Submit to an Edge Compute Node with a Neural Compute Stick 2
We write a script to submit a job to an [IEI Tank 870-Q170](https://software.intel.com/en-us/iot/hardware/iei-tank-dev-kit-core) edge
node with an [Intel Core i5-6500te CPU](https://ark.intel.com/products/88186/Intel-Core-i5-6500TE-Processor-6M-Cache-up-to-3-30-GHz-).
The inference workload should run on an [Intel Neural Compute Stick 2](https://software.intel.com/en-us/neural-compute-stick) installed
in this node.

```sh
vpu_job_id = !qsub queue_job.sh -d . -l nodes=tank-870:i5-6500te:intel-ncs2 -F "[model_path] MYRIAD [original_video_path] /data/queue_param/manufacturing.npy [output_path] 2" -N store_core
```

### Submit to an Edge Compute Node with IEI Mustang-F100-A10
We write a script to submit a job to an [IEI Tank 870-Q170](https://software.intel.com/en-us/iot/hardware/iei-tank-dev-kit-core) edge
node with an [Intel Core™ i5-6500te CPU](https://ark.intel.com/products/88186/Intel-Core-i5-6500TE-Processor-6M-Cache-up-to-3-30-GHz-).
The inference workload will run on the [IEI Mustang-F100-A10 FPGA](https://www.ieiworld.com/mustang-f100/en/) card installed in this
node.

```sh
fpga_job_id = !qsub queue_job.sh -d . -l nodes=1:tank-870:i5-6500te:iei-mustang-f100-a10 -F "[model_path] HETERO:FPGA,CPU [original_video_path] /data/queue_param/manufacturing.npy [output_path] 2" -N store_core
```

### Step 4 - Compare performance
We then compare performance on these devices on these 3 metrics-

* FPS
* Model Load Time
* Inference Time
