# Simple testing script to validate the welcome email functionality
# Replace the email address before running

from modules.email_sender import send_welcome_email

# Replace this with your own email for testing
send_welcome_email("your_email@gmail.com", "Rahul")
