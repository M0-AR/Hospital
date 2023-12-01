import paramiko

# Server credentials and file paths
host = "130.226.25.53"  # Replace with the server's IP address
port = 22
username = "mohash"  # Replace with your username
password = ""  # Replace with your password or use a private key
local_file_path = "./152_test.txt"  # Replace with the path to the local file
remote_file_path = "./tmp/"  # Replace with the target path on the server

# Initialize the SSH client
client = paramiko.SSHClient()

# Optional: automatically accept unknown SSH host keys (do not use in production)
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# Connect to the server
client.connect(hostname=host, port=port, username=username, password=password)

# Initialize SFTP session
sftp = client.open_sftp()

# Upload the file
sftp.put(local_file_path, remote_file_path)

# Close the SFTP session and SSH connection
sftp.close()
client.close()

print("File uploaded successfully.")
