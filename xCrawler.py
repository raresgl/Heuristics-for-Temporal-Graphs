#Instead of using X api (which would cost 200$ a month) I decided to go forward with bluesky
#It uses at protocol so everything I did previously was useless.
import os
from atproto import AtUri, Client, models
from atproto import FirehoseSubscribeReposClient, parse_subscribe_repos_message
from typing import List
import pandas as pd
from datetime import datetime

client = Client()
bskyUsername = os.environ.get("BSKYUSER")
bskyPasswd = os.environ.get("BSKYPASSWORD")
client.login(bskyUsername, bskyPasswd)
# client.send_post(text='API Test: Hello World!')

# posts = client.app.bsky.feed.post.list(client.me.did, limit=10)
# for uri, post in posts.records.items():
#     print(uri, post.text)

# data = client.app.bsky.feed.get_feed_generator({
#     'feed': 'at://did:plc:z72i7hdynmk6r22z27h6tvur/app.bsky.feed.generator/whats-hot'
# })

# view = data.view
# creator = view.creator
# display_name = view.display_name
# like_count = view.like_count
# created_at = view.indexed_at

# print(f"{view} \n")
# print(f"{creator} \n")
# print(f"{display_name} \n")
# print(f"{like_count} \n")


def parse_feed_post(feed_post):
    """Parse a single FeedViewPost object and extract relevant information"""
    post = feed_post.post
    
    # Extract image information if it exists
    images = []
    if hasattr(post, 'embed') and post.embed and hasattr(post.embed, 'images'):
        for img in post.embed.images:
            image_info = {
                'alt': img.alt,
                'thumb': img.thumb if hasattr(img, 'thumb') else None,
                'fullsize': img.fullsize if hasattr(img, 'fullsize') else None
            }
            images.append(image_info)
    
    # Extract mentions from facets
    mentions = []
    if hasattr(post.record, 'facets') and post.record.facets:
        for facet in post.record.facets:
            for feature in facet.features:
                if hasattr(feature, 'did'):  # This is a mention
                    mentions.append(feature.did)
    
    # Build the post data dictionary
    post_data = {
        # 'uri': post.uri,
        # 'cid': post.cid,
        # 'author_handle': post.author.handle,
        'author_display_name': post.author.display_name,
        # 'author_avatar': post.author.avatar,
        'text': post.record.text,
        'created_at': post.record.created_at,
        'indexed_at': post.indexed_at,
        'like_count': post.like_count,
        'reply_count': post.reply_count,
        'repost_count': post.repost_count,
        'quote_count': post.quote_count,
        'languages': post.record.langs if hasattr(post.record, 'langs') else None,
        'mentions': mentions,
        # 'images': images,
    }
    
    return post_data

def process_feed(feed_data: List, save_to_csv=True):
    """Process a list of feed posts and optionally save to CSV"""
    all_posts = []
    
    for feed_post in feed_data:
        post_data = parse_feed_post(feed_post)
        all_posts.append(post_data)
    
    if save_to_csv:
        df = pd.DataFrame(all_posts)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bluesky_feed_posts_{timestamp}.csv"
        
        # Flatten the images and mentions lists to strings for CSV storage
        #df['images'] = df['images'].apply(lambda x: ' | '.join([img['fullsize'] for img in x if img['fullsize']]) if x else '')
        df['mentions'] = df['mentions'].apply(lambda x: ' | '.join(x) if x else '')
        
        df.to_csv(filename, index=False)
        print(f"Saved {len(all_posts)} posts to {filename}")
    
    return all_posts

# Example usage with your current setup:
if __name__ == "__main__":
    data = client.app.bsky.feed.get_feed({
    'feed': 'at://did:plc:z72i7hdynmk6r22z27h6tvur/app.bsky.feed.generator/whats-hot',
    'limit': 30,
    }, headers={'Accept-Language': 'english'})

    feed = data.feed
    print(len(feed))
    next_page = data.cursor

    posts = process_feed(feed)
    
    # Example of accessing parsed data
    for post in posts:
        print(f"\nPost by {post['author_display_name']}:")
        print(f"Text: {post['text']}")
        print(f"Likes: {post['like_count']}")
        # if post['images']:
        #     print(f"Number of images: {len(post['images'])}")
        if post['mentions']:
            print(f"Mentions: {', '.join(post['mentions'])}")