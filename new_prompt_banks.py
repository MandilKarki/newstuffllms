The task you're being asked to do is to classify emails as either "Bank-related" or "Not Bank-related". You'll be doing this in the context of a potential insider threat detection system within the banking and finance sector.

"Bank-related" refers to emails that contain any content, attachments, or signals that relate to banking or financial activities, operations, data, or any sensitive information associated with a banking institution. These can include but are not limited to customer data, financial reports, transaction details, bank account details, and other sensitive banking or financial information.

The 'Red Flag' field plays a crucial role in assessing the nature of the email. If 'Red Flag' is true, it means the email recipient's domain is the same as the sender's. This could imply self-emailing of bank-related information, and hence, it should be scrutinized closely for bank-related content.

On the other hand, if 'Red Flag' is false, indicating the recipient is not the sender themselves, the email might be of lower risk but it still requires careful examination for any bank-related content.

Your objective is to accurately classify emails as either "Bank-related" or "Not Bank-related" based on the above criteria.

Few-shot examples:

Example 1:

Red Flag: True
Email Text: "the private key for our cryptocurrency wallet. Please do not share this with anyone else. It's vital we keep this information secure for the safety of our clients and our institution."
Classification: Bank-related

Example 2:

Red Flag: True
Email Text: "From: Aminul Islam chyaminul@gmail.com To: Islam, Aminul Z aminul.z.islam@[SENSITIVE INFO].com
---------- Forwarded message --------- From: Cheri-Lea Thunack Cheri-Lea.Thunack@toronto.ca Date: Fri, Jun 9, 2023 at 11:00 AM To: chyaminul@gmail.com chyaminul@gmail.com Hello Parent/Caregiver, Welcome to Heron Park Summer Camp and we look forward to seeing you this summer! Please find attached the Welcome Letter, and all the camper forms that need to filled out. Thank you, Heron Park Summer Camp Team☺"
Classification: Not Bank-related

Example 3:

Red Flag: False
Email Text: "Hey, While going through our records, I noticed a discrepancy in our customer data. I cross-referenced it with our previous records and made the necessary changes. Please find attached the corrected file. Best, Sarah."
Classification: Bank-related

Example 4:

Red Flag: True
Email Text: "Hi, I managed to get a hold of the bank's confidential financial reports during my late shift. I can share it with you for a good price. No one will know about it. Let me know if you're interested."
Classification: Bank-related

The actual task will involve:

Red Flag: True
[The Email content goes here]
Classification: [Your model's output: "Bank-related" or "Not Bank-related"]
