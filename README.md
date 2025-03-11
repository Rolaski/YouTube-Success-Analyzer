# YouTube Success Analyzer

## Project Overview
The **YouTube Success Analyzer** is an academic project designed to analyze the success rate of YouTube content creators using graph databases. The system leverages ArangoDB to model relationships between creators, videos, hashtags, and languages, providing insights into what factors contribute to a creator's success.

## Technologies Used
- **Database:** ArangoDB (Graph Database)
- **Backend:** Python
- **Libraries:** `pyArango`, `dotenv`, `csv`
- **Visualization & API:** Streamlit (Planned)
- **DevOps:** Docker (Optional)

## Project Structure
```
project-root/
│
├── data_preparation/
│   ├── data.csv                # Dataset with YouTube data
│   ├── import_data.py          # Script to import data into ArangoDB
│   ├── clearing_db.py          # Script to clear all collections from ArangoDB
│   └── .env-example            # Example environment file
│
└── README.md                  # Project documentation
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
### 1. Setup Environment
- Install ArangoDB and run the server locally.
- Create a database with the desired name.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file based on `.env-example` and set the following variables:
```
ARANGO_USERNAME=your_username
ARANGO_PASSWORD=your_password
ARANGO_DATABASE=your_database_name
```

### 4. Import Data
Navigate to the `data_preparation` folder and run:
```bash
python import_data.py
```

### 5. Clear Database (Optional)
To clear all collections from the database, run:
```bash
python clearing_db.py
```

## Database Structure
- **Creators**: Nodes representing content creators
- **Videos**: Nodes representing YouTube videos
- **HasCreator**: Edges connecting creators to their videos
- **InLanguage**: Edges connecting videos to their language
- **HasHashtag**: Edges connecting videos to hashtags

## Future Features
- Data visualization with Streamlit
- Success rate prediction based on historical data

## Authors
- Jakub Jakubowski 
- [Kacper Bułaś](https://github.com/bolson1313)

## License
This project is developed as part of an academic course on Non-Relational Databases.
