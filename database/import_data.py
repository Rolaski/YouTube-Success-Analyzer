import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from arango import ArangoClient
import re
import hashlib


def import_data_to_arango():
    # Load environment variables
    load_dotenv()

    # Get ArangoDB credentials from environment
    username = os.getenv('ARANGO_USERNAME')
    password = os.getenv('ARANGO_PASSWORD')
    database_name = os.getenv('ARANGO_DATABASE')

    # Connect to ArangoDB
    client = ArangoClient(hosts='http://localhost:8529')

    # Login to ArangoDB
    sys_db = client.db('_system', username=username, password=password)

    # Create database if it doesn't exist
    if not sys_db.has_database(database_name):
        sys_db.create_database(database_name)
        print(f"Created database: {database_name}")

    # Connect to the database
    db = client.db(database_name, username=username, password=password)

    # Get the current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load the normalized data
    normalized_data_path = os.path.join(script_dir, '../data_preparation/normalized_data.csv')
    df = pd.read_csv(normalized_data_path)

    # Create collections if they don't exist
    # 1. Videos collection
    if not db.has_collection('videos'):
        db.create_collection('videos')
        print("Created 'videos' collection")

    # 2. Creators collection
    if not db.has_collection('creators'):
        db.create_collection('creators')
        print("Created 'creators' collection")

    # 3. Edge collection (video_by_creator)
    if not db.has_collection('video_by_creator'):
        db.create_collection('video_by_creator', edge=True)
        print("Created 'video_by_creator' edge collection")

    # Access the collections
    videos_collection = db.collection('videos')
    creators_collection = db.collection('creators')
    edges_collection = db.collection('video_by_creator')

    # Clear collections before import (optional)
    videos_collection.truncate()
    creators_collection.truncate()
    edges_collection.truncate()

    # Helper function to sanitize values for ArangoDB
    def sanitize_for_arango(val):
        if pd.isna(val) or val is None:
            return None
        if isinstance(val, (np.integer, np.floating)):
            return float(val) if isinstance(val, np.floating) else int(val)
        if isinstance(val, bool):
            return val
        return str(val)

    # Helper function to create valid document keys
    def create_valid_key(input_str):
        if not input_str:
            return "unknown_" + hashlib.md5(str(np.random.random()).encode()).hexdigest()[:10]

        # Remove invalid characters and replace with underscores
        key = re.sub(r'[^a-zA-Z0-9_]', '_', str(input_str))

        # Ensure the key starts with a letter
        if not key[0].isalpha():
            key = 'k_' + key

        # Limit key length (ArangoDB has a maximum key length)
        if len(key) > 254:
            key = key[:254]

        return key

    # Process data and insert into collections
    creators_map = {}  # To keep track of creators we've already added
    processed_video_ids = set()  # To keep track of processed video IDs

    for idx, row in df.iterrows():
        try:
            # Create a unique ID for the creator based on channel URL
            channel_url = row['Channel URL']

            # Handle special case where channel URL ends with "- YouTube"
            if isinstance(channel_url, str) and "- YouTube" in channel_url:
                channel_name = channel_url.replace(" - YouTube", "")
                creator_key = create_valid_key(channel_name)
            else:
                # Extract the channel ID or handle from the URL
                if isinstance(channel_url, str):
                    if '/c/' in channel_url:
                        creator_key = channel_url.split('/c/')[-1]
                    elif '/channel/' in channel_url:
                        creator_key = channel_url.split('/channel/')[-1]
                    elif '/user/' in channel_url:
                        creator_key = channel_url.split('/user/')[-1]
                    else:
                        creator_key = channel_url.split('/')[-1]
                else:
                    # If channel_url is not a string, use the creator name or a unique identifier
                    creator_key = create_valid_key(row['Creator Name']) if not pd.isna(
                        row['Creator Name']) else f"creator_{idx}"

            # Ensure creator key is valid
            creator_key = create_valid_key(creator_key)

            # Only add creator if not already added
            if creator_key not in creators_map:
                creator_doc = {
                    '_key': creator_key,
                    'channel_url': sanitize_for_arango(channel_url),
                    'name': sanitize_for_arango(row['Creator Name']),
                    'gender': sanitize_for_arango(row['Creator Gender']),
                    'total_subscribers': sanitize_for_arango(row['Total Channel Subcribers']),
                    'total_views': sanitize_for_arango(row['Total Chanel Views']),
                    'video_count': sanitize_for_arango(row['No of Videos the Channel']),
                    'playlist_count': sanitize_for_arango(row['No of Playlist']),
                    'community_engagement': sanitize_for_arango(row['Community Engagement (Posts per week)'])
                }

                # Ensure all values in the document are valid
                creator_doc = {k: v for k, v in creator_doc.items() if v is not None}

                # Insert creator document
                creators_collection.insert(creator_doc)
                creators_map[creator_key] = True
                print(f"Added creator: {creator_key}")

            # Extract video ID from the URL
            video_url = row['Video Link']
            original_video_id = ""

            if isinstance(video_url, str):
                # Extract the video ID from the URL
                if 'watch?v=' in video_url:
                    original_video_id = video_url.split('watch?v=')[-1]
                    # Remove any additional parameters
                    if '&' in original_video_id:
                        original_video_id = original_video_id.split('&')[0]
                else:
                    # Fallback: use the last part of the URL
                    original_video_id = video_url.split('/')[-1]

            # Ensure we have a valid video ID
            if not original_video_id:
                original_video_id = f"video_{idx}"

            # Create a valid document key
            video_id = create_valid_key(original_video_id)

            # Check if we've already processed this video
            if video_id in processed_video_ids:
                # Make the key unique by adding a suffix
                video_id = f"{video_id}_{idx}"

            processed_video_ids.add(video_id)

            # Create video document
            video_doc = {
                '_key': video_id,
                'url': sanitize_for_arango(video_url),
                'title': sanitize_for_arango(row['Video Title']),
                'views': sanitize_for_arango(row['Video Views']),
                'duration_seconds': sanitize_for_arango(row['Duration in Seconds']),
                'upload_date': sanitize_for_arango(row['Date of Video Upload']),
                'likes': sanitize_for_arango(row['No of Likes']),
                'language': sanitize_for_arango(row['Language of the Video']),
                'subtitle': sanitize_for_arango(row['Subtitle']),
                'description': sanitize_for_arango(row['Video Description']),
                'hashtags': sanitize_for_arango(row['Hashtags']),
                'comment_count': sanitize_for_arango(row['No of Comments']),
                'last_comment_date': sanitize_for_arango(row['Date of the Last Comment']),
                'max_quality': sanitize_for_arango(row['Maximum Quality of the Video']),
                'premiered': sanitize_for_arango(row['Premiered or Not']),
                'data_collector': sanitize_for_arango(row['Intern Who Collected the Data'])
            }

            # Ensure all values in the document are valid
            video_doc = {k: v for k, v in video_doc.items() if v is not None}

            # Insert video document
            videos_collection.insert(video_doc)
            print(f"Added video: {video_id}")

            # Create edge between creator and video
            edge_doc = {
                '_from': f'creators/{creator_key}',
                '_to': f'videos/{video_id}'
            }

            # Insert edge document
            edges_collection.insert(edge_doc)

        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            print(f"Problematic row data: {row}")
            continue

    # Create indexes for better performance
    videos_collection.add_hash_index(['title'])
    videos_collection.add_hash_index(['language'])
    creators_collection.add_hash_index(['name'])

    # Get counts of documents
    video_count = videos_collection.count()
    creator_count = creators_collection.count()
    edge_count = edges_collection.count()

    print(f"Import completed successfully!")
    print(f"Imported {video_count} videos")
    print(f"Imported {creator_count} creators")
    print(f"Created {edge_count} relationships")


if __name__ == "__main__":
    import_data_to_arango()