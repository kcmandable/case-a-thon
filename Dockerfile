# Use an official lightweight Python image
FROM python:3.9-slim

# Create a working directory
WORKDIR /app

# Copy all project files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Run your Dash app
CMD ["python", "app.py"]
