# Script to send update emails to all registered (non-anonymous) users
# Reads from the user_logs.csv and uses send_update_email()

import csv
from modules.email_sender import send_update_email

def notify_all_users():
    sent = 0
    with open("modules/user_logs.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["Name"]
            email = row["Email"]
            if email.lower() != "anonymous":
                try:
                    send_update_email(email, name)  # Send email update
                    sent += 1
                except Exception as e:
                    print(f"❌ Failed to email {email}: {e}")
    print(f"\n📬 Update email sent to {sent} users.")

if __name__ == "__main__":
    notify_all_users()
