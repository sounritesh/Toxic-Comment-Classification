docker build -t gcr.io/overflow-production/pacemaker:v1 .

docker run -it -p 80:80 gcr.io/overflow-production/pacemaker:v1

docker push gcr.io/overflow-production/pacemaker:v1