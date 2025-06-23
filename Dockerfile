# official Python image
FROM python:3.10

# working directory in the container
WORKDIR /app

# copy the files in the container
COPY . .

# install dependencies
RUN pip install -r requirements.txt

# expose port 
EXPOSE 5000

# run the app
CMD [ "python" ,"app.py" ]
