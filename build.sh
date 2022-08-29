docker buildx build --platform=linux/arm64 -t gcr.io/overflow-production/pacemaker:v1-arm64 .

docker push gcr.io/overflow-production/pacemaker:v1-arm64