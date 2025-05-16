import pandas as pd
from langchain_core.documents import Document

def dataconverter():
    product_data = pd.read_csv("data\\flipkart_updated.csv")
    data = product_data[['ProductName', 'Description', 'Ratings', 'Prices']]

    product_list = []

    for index, row in data.iterrows():
        object = {
            "ProductName": row["ProductName"],
            "Description": row["Description"],
            "Ratings": row["Ratings"],
            "Prices": row["Prices"]
        }
        product_list.append(object)

    docs = []

    for object in product_list:
        metadata = {"ProductName": object["ProductName"]}
        page_content = (
            f"{object['ProductName']} - {object['Description']}. "
            f"Rated {object['Ratings']}. Price: {object['Prices']}."
        )
        doc = Document(page_content=page_content, metadata=metadata)
        docs.append(doc)
    return docs
