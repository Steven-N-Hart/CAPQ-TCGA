import argparse
from google.cloud import bigquery
import logging
logger = logging.getLogger()


def get_embedding_length(client, project_id, dataset_id, table_name, embedding_column):
    # SQL query to get the length of the embedding array
    query = f"""
        SELECT ARRAY_LENGTH({embedding_column}) as embedding_length
        FROM `{project_id}.{dataset_id}.{table_name}`
        WHERE ARRAY_LENGTH({embedding_column}) IS NOT NULL
        LIMIT 1
    """

    # Run the query
    query_job = client.query(query)
    result = query_job.result()  # Wait for the query to complete

    # Fetch the first row
    row = next(result)

    return row['embedding_length']



def create_regression_model(project_id, dataset_id, table_name, model_name, label_column, embedding_column, model_type, model_iterations):
    # Initialize the BigQuery client
    client = bigquery.Client(project=project_id)

    # Get the embedding length
    embedding_length = get_embedding_length(client, project_id, dataset_id, table_name, embedding_column)

    # Generate a list of flattened embedding features
    flattened_columns = ', '.join(
        [f"{embedding_column}[OFFSET({i})] AS {embedding_column}_{i}" for i in range(embedding_length)])

    # Define your SQL query to create the regression model
    create_model_query = f"""
        CREATE OR REPLACE MODEL `{project_id}.{dataset_id}.{model_name}`
        OPTIONS(
        model_type='{model_type}',
        input_label_cols=['{label_column}'],
        NUM_TRIALS={model_iterations}
        ) AS
        SELECT
          {flattened_columns},  -- Use the entire embedding array as a feature
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
    parser.add_argument('--embedding_column_name', default='embedding', help="The BigQuery column name containing the embedding data")
    parser.add_argument('--label_column_name', default='er_status_by_ihc', help="The BigQuery column name containing the label", required=True)

    parser.add_argument('--model_name', default=None, help="The name for the model to be created in BigQuery")
    parser.add_argument('--model_type', default='LOGISTIC_REG', help="The type of model to be created.",
                        choices=['LINEAR_REG', 'LOGISTIC_REG'])
    parser.add_argument('--model_iterations', default=5, type=int, help="Number of trials to perform")
    parser.add_argument('--verbosity', help='Logging level',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')
    args = parser.parse_args()
    logger.setLevel(args.verbosity)

    if not args.model_name:
        args.model_name = f"{args.table_name}_{args.label_column_name}_{args.embedding_column_name}_{args.model_type}"


    if args.model_type == 'LINEAR_REG' or args.model_type == 'LOGISTIC_REG':
        create_regression_model(args.project_id, args.dataset_id, args.table_name, args.model_name, args.label_column_name, args.embedding_column_name, args.model_type, args.model_iterations)
    else:
        raise NotImplemented
