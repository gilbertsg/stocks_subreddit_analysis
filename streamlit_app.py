import streamlit as st
import requests
import json

# Title of the page
st.title("📈 r/Stocks vs 💎🤲 r/WallStreetBets Classification")
st.header("This app will predict if a given post comes from r/Stocks or r/WSB, given its post title")
st.caption("For more details, please visit https://github.com/gilbertsg/stocks_subreddit_analysis")
st.header("")
st.subheader("You may want to try the following stereotypical sentences from each subreddit:")
st.caption('- typical r/Stocks sentence: "I am learning to invest in index funds to get good dividends in the long term"')
st.caption('- typical r/WallStreetBets sentence: "Dumping all my life savings to GME tonight. GME TO THE MOON 🚀🚀🚀"')
st.header("")

# Get user inputs
title = st.text_area("📚 Please input the reddit post title:")

# Display the inputs
user_input = {"title":title}
st.write("User input:")
st.write(user_input)

# Code to post the user inputs to the API and get the predictions
# Paste the URL to your GCP Cloud Run API here!
api_url = 'https://subreddit-classification-runrqp42la-as.a.run.app'
api_route = '/predict'

response = requests.post(f'{api_url}{api_route}', json=json.dumps(user_input)) # json.dumps() converts dict to JSON
predictions = response.json()

# Add a submit button
if st.button("Submit"): 
    st.write(f"Prediction: {predictions['predictions'][0]}")
    
st.caption("1 indicates that the post is predicted to be from r/Stocks")
st.caption("0 indicates that the post is predicted to be from r/WallStreetBets")
