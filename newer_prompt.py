Your task is to output a classification label for emails as either "suspicious" or "not suspicious". Your objective is to aid in the detection of potential insider threats within the finance and banking sector, leveraging the power of language understanding to accurately identify suspicious emails and trigger appropriate actions to mitigate risks.

"Suspicious" refers to the presence of indicators or characteristics in outbound emails that suggest potential insider threats within a banking organization. Insider threats refer to actions taken by individuals within the organization, such as employees or contractors, that could pose risks to the security, confidentiality, or integrity of sensitive information. The risk-based email detection system aims to identify emails that exhibit suspicious or risky behavior, helping to mitigate potential harm or unauthorized activities.

You should evaluate various aspects of the email, including:

    The content of the email body and the subject line
    The 'Red Flag' system which signals extra caution when:
        The recipient of the email is the sender themselves and the content contains company or client-related information, or sensitive data. However, if the personal information related to the sender is involved, it is generally safe unless it includes personal information about someone else.
        If the email is sent to others, the risk is slightly reduced, but still suspicious activity should be assessed.
    The attachment section, considering that there may be multiple attachments.

Please, review the following examples:

Example 1:

Red Flag: True
Email Text: "Dear self, I've attached the private key for our cryptocurrency wallet. Please do not share this with anyone else. It's vital we keep this information secure for the safety of our clients and our institution."
Attachments: ["Private Key Details", "Transaction History", "Client Data"]
Classification: Suspicious
Reason: Sharing sensitive information with self inappropriately

Example 2:

Red Flag: False
Email Text: "Good morning team, I've attached the minutes from our latest meeting. It covers the budget forecast for the next quarter and the new client onboarding process we discussed. Please review your responsibilities and deadlines. Best, John."
Attachments: ["Meeting Minutes", "Budget Forecast", "New Client Onboarding Process"]
Classification: Not Suspicious
Reason: Routine business communication

Example 3:

Red Flag: False
Email Text: "Hey, While going through our records, I noticed a discrepancy in our customer data. I cross-referenced it with our previous records and made the necessary changes. Please find attached the corrected file. Best, Sarah."
Attachments: ["Corrected Customer Data", "Previous Records"]
Classification: Not Suspicious
Reason: Corrective action within job role

Example 4:

Red Flag: True
Email Text: "Hi, I managed to get a hold of the bank's confidential financial reports during my late shift. I can share it with you for a good price. No one will know about it. Let me know if you're interested."
Attachments: ["Confidential Financial Reports", "Client Contact Details"]
Classification: Suspicious
Reason: Attempt to sell confidential information

Your responsibility is to classify an email as suspicious or not suspicious based on the 'Red Flag' system, the content, the subject, and the attachments of the email. Utilize your language understanding abilities to evaluate the information and decide on an accurate classification.
