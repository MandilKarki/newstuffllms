# Instruction :

Given the subject line, email body, any attachments, sender domain, sent-to-self flag, access to client data, and attachment similarity flag, classify the email as "Suspicious" or "Not suspicious".

The main task is to detect potential insider threats within the finance and banking sector by analyzing outbound emails. An insider threat refers to harmful or unauthorized actions taken by individuals within the organization. Notably, the outbound emails may not always be drastic in nature but subtle. These subtle signs might include sending client data, project information, or company data to oneself for personal gains.

To classify an email as either "Suspicious" or "Not suspicious", we would follow these steps:

Step 1: Look at the subject line and body for any signs of data exfiltration.

Step 2: Analyze the domain. If the email is sent to personal domains like Gmail or Outlook, there is a higher risk, especially if the sender has access to client data.

Step 3: Check the sent-to-self flag. If this is true, determine the reason behind sending company data to oneself. This can be a potential risk if company policies are violated.

Step 4: Consider if the sender has access to client data. If they do, and they are sending data outside, it could be a significant risk.

Step 5: Check the attachment similarity flag. If the attachments and sender names match, it indicates the document is about them, and it might be less suspicious. Conversely, a mismatch could be indicative of a potential data breach.

### Response

Subject: {subject}

Body: {email}

Attachments: {attachments}

Domain: {domain}

Sent to Self: {sent_to_self_flag}

Access to Client Data: {access_to_client_data}

Attachment Similarity Flag: {attachment_similarity_flag}

# Reasoning

Step 1:

Step 2:

Step 3:

Step 4:

Step 5:


Example 1:

Subject: Client Financial Report

Body: Please check the attached report and let me know.

Attachments: ['financial_report_Q2.pdf']

Domain: gmail.com

Sent to Self: False

Access to Client Data: True

Attachment Similarity Flag: False

# Reasoning

Step 1: The subject suggests it contains client financial information.

Step 2: The email is being sent to a personal domain which heightens the risk.

Step 3: The mail is not sent to self, which makes it more alarming when combined with the nature of the attachment and the domain.

Step 4: The sender has access to client data, which increases the risk.

Step 5: Attachment name and sender name do not match, indicating a potential risk.

# Classification: Suspicious

Example 2:

Subject: Vacation Photos

Body: Attached are the photos from our vacation.

Attachments: ['beach_photo.jpg', 'sunset_view.png']

Domain: outlook.com

Sent to Self: True

Access to Client Data: False

Attachment Similarity Flag: True

# Reasoning

Step 1: The subject and body indicate personal content.

Step 2: Though it's sent to a personal domain, the nature of the content does not suggest any business-related data.

Step 3: The email is sent to self, but the content is personal.

Step 4: The sender does not have access to client data, reducing the risk.

Step 5: While the attachment doesn't strictly have the sender's name, the nature of the attachment aligns with the subject, reducing suspicion.

# Classification: Not Suspicious

Example 3:

Subject: Meeting Agenda

Body: Please find the attached agenda for tomorrow's client meeting.

Attachments: ['client_meeting_agenda.docx']

Domain: businesscorp.com

Sent to Self: False

Access to Client Data: True

Attachment Similarity Flag: False

# Reasoning

Step 1: The subject and body suggest it's a business-related document.

Step 2: The email is sent to a professional domain which seems appropriate.

Step 3: The email isn't sent to self but it appears to be business communication.

Step 4: The sender has access to client data, but in this context, the sharing seems appropriate.

Step 5: Attachment name and sender name do not match, but the attachment seems relevant to the subject.

# Classification: Not Suspicious
