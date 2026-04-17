# syntax=docker/dockerfile:1.7

# Dockerfile for ChatImageJ Vue 3 Frontend
# Multi-stage build for optimized image size

# =============================================================================
# Stage 1: Builder - Build the Vue application
# =============================================================================
FROM node:22-alpine AS builder

RUN npm install -g pnpm
ENV PNPM_HOME=/pnpm \
    PNPM_STORE_DIR=/pnpm/store

# Set working directory
WORKDIR /app

# Copy dependency files (for better layer caching)
COPY package.json pnpm-lock.yaml ./

# Install dependencies
RUN --mount=type=cache,target=/pnpm/store \
    pnpm install --frozen-lockfile

# Copy source files
COPY . .

# Build the application
RUN pnpm run build

# =============================================================================
# Stage 2: Runtime - Serve with nginx
# =============================================================================
FROM nginx:alpine

# Copy built assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Expose port 80
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
