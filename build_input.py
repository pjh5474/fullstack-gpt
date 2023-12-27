import pandas as pd

# Read the Excel file into a Pandas DataFrame
excel_file = "COSMETICS_PMX.xlsx"  # Replace with your file name
data = pd.read_excel(excel_file)

# Define the path for the output text file
output_file = "product_information.txt"

# Open the text file in write mode
with open(output_file, "w") as file:
    # Loop through each row in the DataFrame
    for index, row in data.iterrows():
        # Extract information and write to the text file
        product_info = f"""
The type of Choice-Cosmetics' cosmetics {row['PRODUCT_NAME']} is {row['PRODUCT_TYPE']}.
The main effects of {row['PRODUCT_NAME']} are {row['concerns']}.
The description of {row['PRODUCT_NAME']} is '{row['PRODUCT_DESCRIPTION']}'.
The price of {row['PRODUCT_NAME']} is {row['PRODUCT_PRICE']} won. \n
            """

        file.write(product_info)
