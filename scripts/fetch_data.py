import requests
import logging
import pandas as pd
import os
from datetime import datetime, timedelta
import re
import sqlite3
from pathlib import Path




# Hardcoded values
ACCESS_TOKEN = "EAANLWJGbsYYBOZCFgzTdoqZCZAunaUab2nvBCaaC3koHCVXjh150QMxavFPRRARMdteZC2O719JgtZBiVuXR4eVCCRmnDAijMcpymZC9j5TfKZBbZCZA4POawmpgpn6KmJxEVKiuSITksabCPUaaDuq1doU8cccSvH6lH594fMTMsqiOd5ZBxLmwBRgUA9OdZCRFxYR"
INSTAGRAM_ACCOUNT_ID = "17841406247689288"
BASE_URL = "https://graph.facebook.com/v12.0"
LOG_FILE = "logs/app.log"

# Setup logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(message)s')


def fetch_post_times():
    """
    Fetch post ID and categorize the time of day (morning, afternoon, night)
    for posts from the last 2 years, then save to a CSV file.
    """
    results = []
    url = f"{BASE_URL}/{INSTAGRAM_ACCOUNT_ID}/media"
    two_years_ago = (datetime.now() - timedelta(days=2 * 365)).strftime('%Y-%m-%dT%H:%M:%S')

    params = {
        "fields": "id,timestamp",
        "since": two_years_ago,
        "access_token": ACCESS_TOKEN
    }

    while url:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            for index, post in enumerate(data.get('data', []), start=1):
                timestamp = post.get("timestamp")
                if timestamp:
                    # Extract hour and categorize time of day
                    hour = int(timestamp.split("T")[1].split(":")[0])
                    if 5 <= hour < 12:
                        time_of_day = "morning"
                    elif 12 <= hour < 17:
                        time_of_day = "afternoon"
                    else:
                        time_of_day = "night"

                    results.append({
                        "id": index,
                        "post_id": post.get("id"),
                        "timestamp": timestamp,
                        "time_of_day": time_of_day
                    })
            url = data.get('paging', {}).get('next')  # Handle pagination
            logging.info("Fetched a page of post time data.")
        else:
            logging.error(f"Error fetching post time data: {response.json().get('error', {}).get('message', 'Unknown error')}")
            break

    save_to_csv(results, "data/instagram_post_times.csv")


def fetch_accounts_reached():
    """
    Fetch the post ID and the amount of accounts reached for each post 
    from the last 2 years and save to a CSV file.
    """
    results = []
    url = f"{BASE_URL}/{INSTAGRAM_ACCOUNT_ID}/media"
    two_years_ago = (datetime.now() - timedelta(days=2 * 365)).strftime('%Y-%m-%dT%H:%M:%S')

    params = {
        "fields": "id,timestamp,insights.metric(reach)",
        "since": two_years_ago,
        "access_token": ACCESS_TOKEN
    }

    while url:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            for index, post in enumerate(data.get('data', []), start=1):
                insights = post.get('insights', {}).get('data', [])
                reach = next((insight['values'][0]['value'] for insight in insights if insight['name'] == 'reach'), 0)

                results.append({
                    "id": index,
                    "post_id": post.get("id"),
                    "timestamp": post.get("timestamp"),
                    "accounts_reached": reach
                })
            url = data.get('paging', {}).get('next')  # Handle pagination
            logging.info("Fetched a page of accounts reached data.")
        else:
            logging.error(f"Error fetching accounts reached data: {response.json().get('error', {}).get('message', 'Unknown error')}")
            break

    save_to_csv(results, "data/instagram_accounts_reached.csv")


def fetch_followers_gained():
    """
    Fetch the post ID and the number of followers gained from each post
    in the last 2 years and save to a CSV file.
    """
    results = []
    url = f"{BASE_URL}/{INSTAGRAM_ACCOUNT_ID}/media"
    two_years_ago = (datetime.now() - timedelta(days=2 * 365)).strftime('%Y-%m-%dT%H:%M:%S')

    params = {
        "fields": "id,timestamp,insights.metric(follows)",
        "since": two_years_ago,
        "access_token": ACCESS_TOKEN
    }

    while url:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            for index, post in enumerate(data.get('data', []), start=1):
                insights = post.get('insights', {}).get('data', [])
                followers_gained = next((insight['values'][0]['value'] for insight in insights if insight['name'] == 'follows'), 0)

                results.append({
                    "id": index,
                    "post_id": post.get("id"),
                    "timestamp": post.get("timestamp"),
                    "followers_gained": followers_gained
                })
            url = data.get('paging', {}).get('next')  # Handle pagination
            logging.info("Fetched a page of followers gained data.")
        else:
            logging.error(f"Error fetching followers gained data: {response.json().get('error', {}).get('message', 'Unknown error')}")
            break

    save_to_csv(results, "data/instagram_followers_gained.csv")



def contains_emoji(text):
    """Check if a text contains emojis."""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002700-\U000027BF"  # dingbats
                               u"\U000024C2-\U0001F251"  # enclosed characters
                               "]+", flags=re.UNICODE)
    return bool(emoji_pattern.search(text))


def contains_hashtag(text):
    """Check if a text contains hashtags."""
    return "#" in text


def fetch_captions_with_analysis():
    """
    Fetch captions from Instagram posts from the last 2 years and analyze:
    - If it contains emojis.
    - If it does not contain emojis.
    - If it contains hashtags.
    Save the results to a CSV file.
    """
    results = []
    url = f"{BASE_URL}/{INSTAGRAM_ACCOUNT_ID}/media"
    two_years_ago = (datetime.now() - timedelta(days=2 * 365)).strftime('%Y-%m-%dT%H:%M:%S')

    params = {
        "fields": "id,caption,timestamp",
        "since": two_years_ago,
        "access_token": ACCESS_TOKEN
    }

    while url:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            for index, post in enumerate(data.get('data', []), start=1):
                caption = post.get("caption", "")
                contains_emoji_flag = contains_emoji(caption)
                contains_hashtag_flag = contains_hashtag(caption)

                results.append({
                    "id": index,
                    "post_id": post.get("id"),
                    "caption": caption,
                    "contains_emoji": contains_emoji_flag,
                    "does_not_contain_emoji": not contains_emoji_flag,
                    "contains_hashtag": contains_hashtag_flag
                })
            url = data.get('paging', {}).get('next')  # Handle pagination
            logging.info("Fetched a page of captions data.")
        else:
            logging.error(f"Error fetching captions data: {response.json().get('error', {}).get('message', 'Unknown error')}")
            break

    save_to_csv(results, "data/instagram_captions_analysis.csv")







def fetch_metrics_last_two_years(metric, filename):
    """
    Fetch a specific metric (likes, shares, or comments) for Instagram posts from the last 2 years
    and save to a CSV file in the data folder.
    """
    results = []
    url = f"{BASE_URL}/{INSTAGRAM_ACCOUNT_ID}/media"
    two_years_ago = (datetime.now() - timedelta(days=2 * 365)).strftime('%Y-%m-%dT%H:%M:%S')

    params = {
        "fields": f"id,timestamp,insights.metric({metric})",
        "since": two_years_ago,
        "access_token": ACCESS_TOKEN
    }

    while url:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            for index, post in enumerate(data.get('data', []), start=1):
                insights = post.get('insights', {}).get('data', [])
                value = next((insight['values'][0]['value'] for insight in insights if insight['name'] == metric), 0)
                results.append({
                    "id": index,
                    "post_id": post.get("id"),
                    metric: value,
                    "timestamp": post.get("timestamp")
                })
            url = data.get('paging', {}).get('next')  # Handle pagination
            logging.info(f"Fetched a page of {metric} data.")
        else:
            logging.error(f"Error fetching {metric} data: {response.json().get('error', {}).get('message', 'Unknown error')}")
            break

    save_to_csv(results, filename)




def save_to_csv(data, filename="data/instagram_post_details.csv"):
    """Save post details to a CSV file."""
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

def fetch_likes():
    """Fetch likes for posts from the last 2 years and save to CSV."""
    fetch_metrics_last_two_years("likes", "data/instagram_likes.csv")


def fetch_shares():
    """Fetch shares for posts from the last 2 years and save to CSV."""
    fetch_metrics_last_two_years("shares", "data/instagram_shares.csv")


def fetch_comments():
    """Fetch comments for posts from the last 2 years and save to CSV."""
    fetch_metrics_last_two_years("comments", "data/instagram_comments.csv")












    
















    


    



