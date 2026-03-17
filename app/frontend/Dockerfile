FROM node:20-alpine AS builder

WORKDIR /app

COPY app/frontend/package.json ./
RUN npm install

COPY app/frontend/ ./
RUN npm run build

# Production image — serve with nginx
FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html

# Proxy /api → backend
COPY app/frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
