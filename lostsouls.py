res = generate_text('''

    # Question: 

    Given the subject line, email content, any attachments, and a 'Sent Self' indicator, how would you classify the email as "Suspicious" or "Not suspicious"?

    # Answer:

    To classify an email as either "Suspicious" or "Not suspicious", we would follow these steps:

    Step 1: Understand the Context:
    We're trying to detect potential insider threats within the finance and banking sector by analyzing outbound emails. An insider threat refers to harmful or unauthorized actions taken by individuals within the organization. Notably, the outbound emails may not always be drastic in nature but subtle. The subtle ones might include sending client data, project information or company data to oneself for personal gains.

    Step 2: Analyze the 'Sent Self' Indicator:
    A 'Sent Self' indicator can mean that the sender of the email is the recipient as well. If it is True, we proceed with extra caution.

    Step 3: Inspect the Email Content and Subject:
    We analyze the content and subject of the email for signs of insider threats. If the content or subject includes sensitive data, client-related information or company-related information, it's a potential threat and should be marked 'Suspicious'. Personal emails which may not be suspicious could include things like personal photos, tax information, event planning, etc. However, if these personal emails include information or documents related to someone else, they may be deemed suspicious.

    Step 4: Consider the Attachments:
    Any attachments to the email should also be scrutinized for potentially sensitive data or suspicious content. Personal attachments like CVs are generally not suspicious unless they are about someone else, which could indicate a data breach.

    Step 5: Analyze Emails with 'Sent Self' as False:
    If the 'Sent Self' is False, it means the recipient of the email is not the sender. These emails may be of lower risk, but we still analyze for signs of suspicious activity. 

    ## Examples:

    # Question 1:

    Subject: Cryptocurrency Wallet Key
    Sent Self: True
    Email Text: "the private key for our cryptocurrency wallet. Please do not share this with anyone else.
    It's vital we keep this information secure for the safety of our clients and our institution."
    Attachment: WalletKey.txt

    # Answer 1:

    Classification: Suspicious

    # Reasoning:

    This email contains sensitive data related to the cryptocurrency wallet of the institution, which is an indication of an insider threat. The subject of the email also refers to a key for a cryptocurrency wallet, further raising suspicions. The attachment seems to be the wallet key, adding to the risk.

    # Question 2:

    Subject: Summer Camp Forms
    Sent Self: True
    Email Text: "From: Aminul Islam chyaminul@gmail.com To: Islam, Aminul Z aminul.z.islam@[SENSITIVE INFO].com  
    ---------- Forwarded message --------- From: Cheri-Lea Thunack Cheri-Lea.Thunack@toronto.ca Date: Fri, Jun 9, 2023 at 11:00 AM To: chyaminul@gmail.com chyaminul@gmail.com  Hello Parent/Caregiver,  Welcome to Heron Park Summer Camp and we look forward to seeing you this summer! Please find attached the Welcome Letter,
    and all the camper forms that need to filled out.  Thank you, Heron Park Summer Camp Team☺"
    Attachment: CamperForms.pdf

    # Answer 2:

    Classification: Not Suspicious

    # Reasoning:

    This email is personal in nature, with the subject suggesting that it's related to summer camp forms. The content of the email does not indicate any signs of an insider threat, and the attachment appears to be related to the summer camp as well, reducing suspicions.

    # Question 3:

    Subject: Updated Customer Data
    Sent Self: False
    Email Text: "Hi team, While going through our records, I noticed a discrepancy in our customer data. I cross-referenced
    it with our previous records and made the necessary changes. Please find attached the corrected file. Best, Sarah."
    Attachment: Corrected_Customer_Data.xlsx

    # Answer 3:

    Classification: Not Suspicious

    # Reasoning:

    The email indicates that the sender is taking proactive steps to correct discrepancies in customer data, which is a legitimate business activity. The attachment appears to be a regular business document, and the 'Sent Self' flag is False, further reducing suspicion.

    # Question 4:

    Subject: Confidential Report
    Sent Self: True
    Email Text: "Hi, I managed to get a hold of the bank's confidential financial reports during my late shift. I can share
    it with you for a good price. No one will know about it. Let me know if you're interested."
    Attachment: Confidential_Report.pdf

    # Answer 4:

    Classification: Suspicious

    # Reasoning:

    This email discusses sharing confidential financial reports for a price, which is a strong indicator of an insider threat. The subject and the attachment further raise suspicions as they deal with a 'Confidential Report'.

''', max_length=1500, temperature=0.3)
print(res[0]["generated_text"])
