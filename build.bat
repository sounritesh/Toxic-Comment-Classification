docker build -t asia.gcr.io/overflow-production/pacemaker:v1 .

docker run -it -p 80:80 asia.gcr.io/overflow-production/pacemaker:v1

docker push asia.gcr.io/overflow-production/pacemaker:v1