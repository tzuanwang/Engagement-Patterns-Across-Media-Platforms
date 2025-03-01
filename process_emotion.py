import argparse
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, FloatType
from transformers import pipeline
import logging
from itertools import islice

logging.basicConfig(level=logging.INFO)

emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def load_model_and_predict(iterator):
    classifier = pipeline(
        "text-classification",
        model="bhadresh-savani/albert-base-v2-emotion",
        return_all_scores=True,
        truncation=True,
    )
    for row in iterator:
        text = row["review_detail"]
        if not text:  # Handle empty or null values gracefully
            yield row
            continue
        try:
            truncated_text = text[:512]

            predictions = classifier(truncated_text, return_all_scores=True)[0]

            emotion_dict = {pred["label"]: float(pred["score"]) for pred in predictions}

            max_emotion = max(predictions, key=lambda x: x["score"])

            new_row = row.asDict()

            new_row["predicted_emotion"] = predictions

            for label in emotion_labels:
                new_row[label] = emotion_dict.get(label, 0.0)

            new_row["emotion"] = max_emotion["label"]

            yield Row(**new_row)

        except Exception as e:
            logging.error(f"Error processing text: {text[:100]}...")
            logging.error(f"Error: {e}")
            new_row = row.asDict()
            new_row["predicted_emotion"] = [{"label": "error", "score": 0.0}]
            for label in emotion_labels:
                new_row[label] = 0.0  # Default scores to 0.0
            new_row["emotion"] = "error"  # Default emotion name
            yield Row(**new_row)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Emotion processing with Spark and Hugging Face"
    )
    parser.add_argument(
        "--input", required=True, help="Path to input data (parquet format)"
    )
    parser.add_argument(
        "--output", required=True, help="Path to output data (parquet format)"
    )
    parser.add_argument(
        "--partitions",
        type=int,
        default=1000,
        help="Number of partitions for input data",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of rows per partition to process (omit to process all)",
    )
    args = parser.parse_args()

    spark = SparkSession.builder.appName("EmotionProcessing").getOrCreate()

    logging.info(f"Reading input data from {args.input}")
    df = spark.read.parquet(args.input).dropna()
    df = df.repartition(args.partitions)

    if args.limit is not None:
        logging.info(f"Processing with a limit of {args.limit} rows per partition")
        limited_rdd = df.rdd.mapPartitions(
            lambda partition: islice(partition, args.limit)
        )
        predicted_df = limited_rdd.mapPartitions(load_model_and_predict).toDF()
    else:
        logging.info("Processing all rows per partition (no limit)")
        predicted_df = df.rdd.mapPartitions(load_model_and_predict).toDF()

    logging.info(f"Writing output data to {args.output}")
    predicted_df.write.mode("overwrite").parquet(args.output)

    spark.stop()
