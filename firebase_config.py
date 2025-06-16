import firebase_admin
from firebase_admin import credentials, firestore
import streamlit as st
import json

if not firebase_admin._apps:
    try:
        firebase_info = json.loads(st.secrets["firebase_key"])
        cred = credentials.Certificate(firebase_info)
    except:
        cred = credentials.Certificate("serviceAccountKey.json")

    firebase_admin.initialize_app(cred)

db = firestore.client()
