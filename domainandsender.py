import difflib

def get_email_domain(email):
    return email.split('@')[1]

def is_sent_to_competitor(sender_email, receiver_emails, competitor_domains):
    sender_domain = get_email_domain(sender_email)
    receiver_domains = [get_email_domain(email) for email in receiver_emails]

    for domain in receiver_domains:
        if domain in competitor_domains:
            return True

    for email in receiver_emails:
        receiver_domain = get_email_domain(email)
        if difflib.SequenceMatcher(None, sender_domain, receiver_domain).ratio() > 0.8:
            return False

    sender_username = sender_email.split('@')[0]
    for email in receiver_emails:
        receiver_username = email.split('@')[0]
        if difflib.SequenceMatcher(None, sender_username, receiver_username).ratio() > 0.8:
            return False

    return False


import csv
import difflib

def is_sent_to_competitor(sender_email, receiver_emails, competitor_domains):
    # Check if the sender email domain is in the competitor domains
    sender_domain = sender_email.split('@')[1].lower()
    if sender_domain in competitor_domains:
        return True
    
    # Check if any of the receiver emails are in the competitor domains
    for receiver_email in receiver_emails:
        receiver_domain = receiver_email.split('@')[1].lower()
        if receiver_domain in competitor_domains:
            return True
    
    return False

# Rest of the code remains the same...
