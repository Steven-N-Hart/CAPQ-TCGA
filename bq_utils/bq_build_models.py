import argparse
from google.cloud import bigquery
import logging
logger = logging.getLogger()

def create_logistic_regression_model(project_id, dataset_id, table_name, model_name):
    # Initialize the BigQuery client
    client = bigquery.Client(project=project_id)

    # Define your SQL query to create the logistic regression model
    create_model_query = f"""
    CREATE OR REPLACE MODEL `{project_id}.{dataset_id}.{model_name}`
    OPTIONS(model_type='logistic_reg') AS
    WITH flattened_data AS (
      SELECT
        ARRAY_TO_STRING(ARRAY_AGG(CAST(embedding AS STRING)), ',') AS flattened_embedding,
        ER_Status
      FROM
        `{project_id}.{dataset_id}.{table_name}`
      GROUP BY
        ER_Status
    )
    SELECT
      CAST(SPLIT(flattened_embedding, ',') AS FLOAT64) AS embedding_features,
      ER_Status
    FROM
      flattened_data;
    """

    # Run the query to create the model
    query_job = client.query(create_model_query)

    # Wait for the query to finish
    query_job.result()

    print("Logistic regression model created successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a logistic regression model in BigQuery.")

    parser.add_argument('--project_id', required=True, help="Your Google Cloud project ID")
    parser.add_argument('--dataset_id', required=True, help="The BigQuery dataset ID")
    parser.add_argument('--table_name', required=True, help="The BigQuery table name containing the data")
    parser.add_argument('--model_name', required=True, help="The name for the model to be created in BigQuery")
    parser.add_argument('--model_type', required=True, help="The type of model to be created.",
                        choices=['LINEAR_REG', 'LOGISTIC_REG'])
    parser.add_argument('--verbosity', help='Logging level',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')
    args = parser.parse_args()

    if args.model_type == 'LINEAR_REG':
        raise NotImplemented
    elif args.model_type == 'LOGISTIC_REG':
        create_logistic_regression_model(args.project_id, args.dataset_id, args.table_name, args.model_name)
    else:
        raise NotImplemented
