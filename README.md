# YouTube Success Analyzer

## Project Overview
The **YouTube Success Analyzer** is an academic project designed to analyze the success rate of YouTube content creators using graph databases. The system leverages ArangoDB to model relationships between creators, videos, hashtags, and languages, providing insights into what factors contribute to a creator's success.

## Prerequisites
- Docker installed on your system
- Python 3.8 or higher
- pip package manager

## Technologies Used
- **Database:** ArangoDB (Graph Database)
- **Backend:** Python
- **Libraries:** 
  - `python-arango` - ArangoDB Python driver
  - `pandas` - Data manipulation
  - `streamlit` - Web interface
  - `scikit-learn` - Machine learning models
  - `plotly` - Data visualization
  - `networkx` - Graph visualization
  - `python-dotenv` - Environment configuration
- **DevOps:** Docker


## Project Structure
```
project-root/
│
├── data_preparation/
│   ├── data.csv                # Dataset with YouTube data
│   ├── data_normalization.py   # Data normalization script
│   ├── graph.py                # Graph visualization module
│   └── normalized_data.csv     # Processed dataset
│
├── database/
│   ├── crud.py                # CRUD operations for ArangoDB
│   ├── import_data.py         # Data import script
│   ├── clearing_db.py         # Database cleanup script
│   ├── validate_import.py     # Import validation script
│   └── .env                   # Database configuration
│
├── models/
│   ├── model.py               # Machine learning model implementation
│   └── youtube_success_model.pkl  # Trained model file
│
├── results/
│   ├── actual_vs_predicted.png  # Model performance visualization
│   └── feature_importance.png   # Feature importance plot
│
├── main.py                     # Main application file
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Dataset
The dataset used in this project is [YouTube Channel and Influencer Analysis](https://www.kaggle.com/datasets/kathir1k/youtube-influencers-data), containing information such as:

- Video Link
- Video Views
- Video Title
- Channel URL
- Creator Name
- Creator Gender
- Total Channel Subscribers
- Total Channel Views
- Duration of Video
- Duration in Seconds
- Date of Video Upload
- No of Likes
- Language of the Video
- Subtitle
- Video Description
- Hashtags
- No of Comments
- Date of the Last Comment
- Maximum Quality of the Video
- No of Videos the Channel
- No of Playlist
- Premiered or Not
- Community Engagement (Posts per week)
- Intern Who Collected the Data


## How to Run the Project
### 1. Docker Setup
```bash
# Pull and run ArangoDB container
docker run -e ARANGO_ROOT_PASSWORD=root -p 8529:8529 -d --name arangodb arangodb:latest

# Verify container is running
docker ps
```

### 2. Database Setup
1. Access ArangoDB web interface at `http://localhost:8529`
2. Login with default credentials:
   - Username: root
   - Password: root
3. Create a new database named `youtube_analysis`

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the `database/` folder with:
```
ARANGO_USERNAME=root
ARANGO_PASSWORD=root
ARANGO_DATABASE=youtube_analysis
```

### 5. Import Data
```bash
cd data_preparation
python import_data.py
```
### 6. Run the Application
```bash
streamlit run main.py
```
The application will be available at `http://localhost:8501`

## Database Structure
- **creators**: Collection storing content creator information
- **videos**: Collection containing YouTube video details
- **video_by_creator**: Edge collection connecting videos to their creators

### creators
- Creator name
- Channel URL
- Total subscribers
- Total channel views
- Channel statistics

### videos
- Video title
- Video views
- Duration
- Upload date
- Likes count
- Comments count
- Video description
- Language
- Quality metrics

### video_by_creator
- Edge collection linking videos to their creators
- Contains references to both creator and video documents
- Represents the ownership relationship between creators and their content

## Future Features
- Data visualization with Streamlit
- Success rate prediction based on historical data

## Authors
- Jakub Jakubowski 
- [Kacper Bułaś](https://github.com/bolson1313)

## License
This project is developed as part of an academic course on Non-Relational Databases.
