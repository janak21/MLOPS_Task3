import smtplib 
  
# creates SMTP session 
s = smtplib.SMTP('smtp.gmail.com', 587) 
  
# start TLS for security 
s.starttls() 
  
# Authentication 
s.login("youremail@gmai.com", "Password of your email") 
  
# message to be sent 
message = "Model is created!"
  
# sending the mail 
s.sendmail("youremail@gmail.com", "receiver@gmail.com", message) 
  
# terminating the session 
s.quit()
