from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import psycopg2

# Configuration
username = 'postgres'
password = 'password123'
host = 'localhost'
port = '5432'
database = 'faceid'

# Create the database URL
database_url = f"postgresql://{username}:{password}@{host}:{port}/{database}"

# Create the Flask application
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ECHO'] = True  # Enable SQL command logging

app.config["JWT_SECRET_KEY"] = "super-secret"

# Initialize SQLAlchemy
db = SQLAlchemy(app)

conn = psycopg2.connect(    
    host=host,
    database=database,
    user=username,
    password=password
)
