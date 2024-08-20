import argparse
from google.cloud import bigquery
import logging
logger = logging.getLogger()

def create_regression_model(project_id, dataset_id, table_name, model_name, label_column, embedding_column, model_type):
    # Initialize the BigQuery client
    client = bigquery.Client(project=project_id)

    # Define your SQL query to create the regression model
    create_model_query = f"""
        CREATE OR REPLACE MODEL `{project_id}.{dataset_id}.{model_name}.{table_name}`
        OPTIONS(
        model_type='{model_type}',
        input_label_cols=['{label_column}']
        ) AS
        SELECT
          {embedding_column},  -- Use the entire embedding array as a feature
          {label_column}
        FROM
          `{project_id}.{dataset_id}.{table_name}`
        WHERE
          ARRAY_LENGTH({embedding_column}) > 0
          AND {embedding_column} IS NOT NULL
          AND {label_column} IS NOT NULL
    """

    # Run the query to create the model
    query_job = client.query(create_model_query)

    # Wait for the query to finish
    query_job.result()

    logger.info("Regression model created successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a logistic regression model in BigQuery.")

    parser.add_argument('--project_id', default='correlation-aware-pq', help="Your Google Cloud project ID")
    parser.add_argument('--dataset_id', default='tcga', help="The BigQuery dataset ID")
    parser.add_argument('--table_name', default='clinical_data', help="The BigQuery table name containing the data")
    parser.add_argument('--embedding_column_name', default='embedding', help="The BigQuery table name containing the data")
    parser.add_argument('--label_column_name', default='er_status_by_ihc', help="The BigQuery table name containing the data")

    parser.add_argument('--model_name', default=None, help="The name for the model to be created in BigQuery")
    parser.add_argument('--model_type', default='LOGISTIC_REG', help="The type of model to be created.",
                        choices=['LINEAR_REG', 'LOGISTIC_REG'])
    parser.add_argument('--verbosity', help='Logging level',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')
    args = parser.parse_args()
    logger.setLevel(args.verbosity)

    if not args.model_name:
        args.model_name = f"{args.model_type}_{args.label_column_name}_{args.embedding_column_name}"


    if args.model_type == 'LINEAR_REG' or args.model_type == 'LOGISTIC_REG':
        create_regression_model(args.project_id, args.dataset_id, args.table_name, args.model_name, args.label_column_name, args.embedding_column_name, args.model_type)
    else:
        raise NotImplemented
