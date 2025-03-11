import os
import pandas as pd
from dotenv import load_dotenv
from arango import ArangoClient


def validate_data():
    # Load environment variables
    load_dotenv()

    # Get ArangoDB credentials from environment
    username = os.getenv('ARANGO_USERNAME')
    password = os.getenv('ARANGO_PASSWORD')
    database_name = os.getenv('ARANGO_DATABASE')

    # Connect to ArangoDB
    client = ArangoClient(hosts='http://localhost:8529')

    # Connect to the database
    db = client.db(database_name, username=username, password=password)

    # Get the current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load the normalized data
    normalized_data_path = os.path.join(script_dir, '../data_preparation/normalized_data.csv')
    df = pd.read_csv(normalized_data_path)

    # Access collections
    videos_collection = db.collection('videos')
    creators_collection = db.collection('creators')
    edges_collection = db.collection('video_by_creator')

    # Get counts
    video_count = videos_collection.count()
    creator_count = creators_collection.count()
    edge_count = edges_collection.count()

    print(f"Database counts:")
    print(f"- Videos: {video_count}")
    print(f"- Creators: {creator_count}")
    print(f"- Relationships: {edge_count}")

    # Check if videos count matches the dataframe
    csv_video_count = len(df)
    print(f"\nCSV data counts:")
    print(f"- Total rows in CSV: {csv_video_count}")

    # Count unique creators in the dataframe
    unique_creators_in_csv = df['Channel URL'].nunique()
    print(f"- Unique creators in CSV (by Channel URL): {unique_creators_in_csv}")

    # Look for creators that might have been missed
    if video_count < csv_video_count:
        print(f"\nWarning: {csv_video_count - video_count} videos from the CSV might not have been imported")

    if creator_count < unique_creators_in_csv:
        print(
            f"\nWarning: {unique_creators_in_csv - creator_count} unique creators from the CSV might not have been imported")

    # Check for videos without creators (orphaned videos)
    aql_orphaned_videos = """
    FOR v IN videos
        LET edges = (FOR e IN video_by_creator FILTER e._to == CONCAT('videos/', v._key) RETURN e)
        FILTER LENGTH(edges) == 0
        RETURN v._key
    """
    orphaned_videos = list(db.aql.execute(aql_orphaned_videos))

    if orphaned_videos:
        print(f"\nFound {len(orphaned_videos)} videos without creators (orphaned videos)")
        print(f"First 10 orphaned videos: {orphaned_videos[:10]}")
    else:
        print("\nNo orphaned videos found - all videos are connected to creators")

    # Check for creators without videos
    aql_creators_without_videos = """
    FOR c IN creators
        LET edges = (FOR e IN video_by_creator FILTER e._from == CONCAT('creators/', c._key) RETURN e)
        FILTER LENGTH(edges) == 0
        RETURN c._key
    """
    creators_without_videos = list(db.aql.execute(aql_creators_without_videos))

    if creators_without_videos:
        print(f"\nFound {len(creators_without_videos)} creators without videos")
        print(f"First 10 creators without videos: {creators_without_videos[:10]}")
    else:
        print("\nNo creators without videos found - all creators have at least one video")

    # Check creator with most videos
    aql_top_creators = """
    FOR c IN creators
        LET video_count = LENGTH(
            FOR v IN 1..1 OUTBOUND CONCAT('creators/', c._key) video_by_creator
            RETURN v
        )
        SORT video_count DESC
        LIMIT 5
        RETURN {
            creator: c.name,
            key: c._key,
            video_count: video_count
        }
    """
    top_creators = list(db.aql.execute(aql_top_creators))

    print("\nTop 5 creators by number of videos:")
    for i, creator in enumerate(top_creators, 1):
        print(f"{i}. {creator['creator']} (ID: {creator['key']}): {creator['video_count']} videos")


if __name__ == "__main__":
    validate_data()