# CAPQ-TCGA

## Part 1. Data Selection

Use MinPath strategy to extract most relevant patches.  First, get a datatable from the Imaging Data Commons:
```shell
python utils/bq_to_gcs.py -h 

usage: bq_to_gcs.py [-h] [--gcp_project_id GCP_PROJECT_ID] [--bucket_name BUCKET_NAME] [--query_file QUERY_FILE] [--workspace_dir WORKSPACE_DIR] [--output_filename OUTPUT_FILENAME] [--verbosity {DEBUG,INFO,WARNING,ERROR,CRITICAL}]

Run a BigQuery query and save results to GCS.

options:
  -h, --help            show this help message and exit
  --gcp_project_id GCP_PROJECT_ID
                        Google Cloud Project ID
  --bucket_name BUCKET_NAME
                        Google Cloud Storage bucket name
  --query_file QUERY_FILE
                        Path to the file containing the BigQuery SQL query
  --workspace_dir WORKSPACE_DIR
                        Directory in the bucket to save the results
  --output_filename OUTPUT_FILENAME
                        Name of the output CSV file
  --verbosity {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Logging level

```
It has the following columns.  The ones in bold are used in later code, so make sure they are present!
* NumFrames
* **SeriesInstanceUID**
* **StudyInstanceUID**
* **SOPInstanceUID**
* ContainerIdentifier
* PatientID
* **TotalPixelMatrixColumns**
* **TotalPixelMatrixRows**
* collection_id
* crdc_instance_uuid
* **gcs_url**
* gcs_bucket
* **pixel_spacing**
* compression
* item_name
* item_value
* **row_num_asc**
* **row_num_desc**

Now, use this file to write image frames using the MiniPath sampling strategy
```shell
python utils/dcm_to_img_frames_job.py -h 

usage: dcm_to_img_frames_job.py [-h] [--gcp_project_id GCP_PROJECT_ID] [--bq_results_csv BQ_RESULTS_CSV] [--bucket_name BUCKET_NAME] [--dirname DIRNAME] [--location LOCATION] [--display_name DISPLAY_NAME]
                                [--container_uri CONTAINER_URI] [--staging_bucket STAGING_BUCKET] [--machine_type MACHINE_TYPE] [--replica_count REPLICA_COUNT]

Process DICOM files and upload images to Google Cloud Storage.

options:
  -h, --help            show this help message and exit
  --gcp_project_id GCP_PROJECT_ID
                        Google Cloud Project ID
  --bq_results_csv BQ_RESULTS_CSV
                        Path to the CSV file containing BQ results.
  --bucket_name BUCKET_NAME
                        Google Cloud Storage bucket name
  --dirname DIRNAME     Directory name for storing images.
  --location LOCATION   Google Cloud Storage bucket location
  --display_name DISPLAY_NAME
                        Display name for Vertex AI job
  --container_uri CONTAINER_URI
                        Display name for VertexAI job
  --staging_bucket STAGING_BUCKET
                        Staging bucket
  --machine_type MACHINE_TYPE
                        Machine type
  --replica_count REPLICA_COUNT
                        Number of replicas
```
## Part 2. Compute Embeddings an targeted patches
Now that a GCS Bucket exists with PNG files, we can iterate through each of them, create and embedding, and store 
the results in a BigQuery Table.

```shell

python utils/image_embedding_job.py -h

usage: image_embedding_job.py [-h] [--gcp_project_id GCP_PROJECT_ID] [--bucket_name BUCKET_NAME] [--folder_prefix FOLDER_PREFIX] [--dataset_name DATASET_NAME] [--table_name TABLE_NAME] [--model_name MODEL_NAME]
                              [--location LOCATION] [--display_name DISPLAY_NAME] [--container_uri CONTAINER_URI] [--staging_bucket STAGING_BUCKET]

Process images and store embeddings in BigQuery.

options:
  -h, --help            show this help message and exit
  --gcp_project_id GCP_PROJECT_ID
                        Google Cloud Project ID
  --bucket_name BUCKET_NAME
                        Google Cloud Storage bucket name
  --folder_prefix FOLDER_PREFIX
                        Folder prefix in the bucket to look for images
  --dataset_name DATASET_NAME
                        BigQuery Dataset Name
  --table_name TABLE_NAME
                        BigQuery Table Name
  --model_name MODEL_NAME
                        BigQuery Dataset Name
  --location LOCATION   Google Cloud Storage bucket location
  --display_name DISPLAY_NAME
                        Display name for VertexAI job
  --container_uri CONTAINER_URI
                        Display name for VertexAI job
  --staging_bucket STAGING_BUCKET
                        Staging bucket
```

## Part 3. Create a model
For simplicity, we are creating a logistic regression model to predict label, given embeddings.
```shell
python bq_utils\bq_build_models.py --label_column_name er_status_by_ihc
python bq_utils\bq_build_models.py --label_column_name pr_status_by_ihc
python bq_utils\bq_build_models.py --label_column_name HER2_newly_derived
python bq_utils\bq_build_models.py --label_column_name Triple_Negative_Status
python bq_utils\bq_build_models.py --label_column_name PAM50_and_Claudin_low__CLOW__Molecular_Subtype
```