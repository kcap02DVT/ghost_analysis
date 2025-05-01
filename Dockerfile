# Build stage
FROM node:20-alpine

WORKDIR /app

# Copy files
COPY . /app/frontend

WORKDIR /app/frontend/project

# Install dependencies
RUN npm install
RUN npm install @headlessui/react react lucide-react @types/react

# Start the application
CMD ["npm", "start"]
