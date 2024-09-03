import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("adql-firestore-firebase-adminsdk-5gujm-bf3bee582a.json")
firebase_admin.initialize_app(cred)

db = firestore.client()