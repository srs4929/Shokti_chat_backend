import os
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import JSONLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from numpy import record

load_dotenv()

file_path = "./data/user_001.json"

def transform_record_to_document(record):
    """Transform a numeric record into a text document with metadata"""

    power_level = "very high" if record['Global_active_power'] >= 3.0 else \
                 "high" if record['Global_active_power'] >= 2.0 else \
                 "medium" if record['Global_active_power'] >= 1.0 else "low"
                 
    time_period = "night" if record['Hour'] < 6 else \
                 "morning" if record['Hour'] < 12 else \
                 "afternoon" if record['Hour'] < 18 else "evening"
    
    day_type = "weekend" if record['Day_of_week'] in [5, 6] else "weekday"

    text_content = f"""
    Power consumption record at {record['Datetime']}:
    - Global Active Power: {record['Global_active_power']:.2f} kW
    - Global Reactive Power: {record['Global_reactive_power']:.2f} kVAR
    - Voltage: {record['Voltage']:.1f} V
    - Global Intensity: {record['Global_intensity']:.1f} A
    - Sub-metering 1 (kitchen): {record['Sub_metering_1']:.1f} kWh
    - Sub-metering 2 (laundry): {record['Sub_metering_2']:.1f} kWh
    - Sub-metering 3 (HVAC): {record['Sub_metering_3']:.1f} kWh
    - Time: {record['Hour']}:00 hours
    - Day of week: {record['Day_of_week']} ({'Weekend' if record['Day_of_week'] in [5, 6] else 'Weekday'})
    - Holiday: {'Yes' if record['Holiday'] else 'No'}

    Summary: {power_level} electricity consumption of {record['Global_active_power']:.3f} kW during {time_period} hours on a {day_type}.
    """
    

    metadata = {
        'datetime': record['Datetime'],
        'global_active_power': float(record['Global_active_power']),
        'global_reactive_power': float(record['Global_reactive_power']),
        'voltage': float(record['Voltage']),
        'global_intensity': float(record['Global_intensity']),
        'sub_metering_1': float(record['Sub_metering_1']),
        'sub_metering_2': float(record['Sub_metering_2']),
        'sub_metering_3': float(record['Sub_metering_3']),
        'hour': int(record['Hour']),
        'day_of_week': int(record['Day_of_week']),
        'holiday': bool(record['Holiday']),
        'is_weekend': record['Day_of_week'] in [5, 6],
        'power_level': power_level,
        'time_period': time_period,
        'day_type': day_type
    }
    
    return Document(page_content=text_content, metadata=metadata)

def rag_pipeline():
    loader = JSONLoader(
        file_path=file_path,
        jq_schema=".[]",
        text_content=False,  
    )

    raw_documents = loader.load()


    documents = []
    for doc in raw_documents:
        record = json.loads(doc.page_content)  
        transformed_doc = transform_record_to_document(record)
        documents.append(transformed_doc)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


    vector_db = FAISS.from_documents(documents, embeddings)
    vector_db.save_local("./data/vector_index/user_001_faiss")

    print("Vector database created and saved successfully!")
    return vector_db

if __name__ == "__main__":
    rag_pipeline()