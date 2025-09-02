#!/bin/bash
# Wait for services to be ready before proceeding with initialization

set -e

echo "Waiting for services to be ready..."

# Function to wait for a service
wait_for_service() {
    local service=$1
    local check_command="$2"
    local timeout=${3:-60}
    local interval=${4:-2}
    
    echo "Waiting for $service to be ready..."
    local count=0
    
    while [ $count -lt $timeout ]; do
        if eval "$check_command" >/dev/null 2>&1; then
            echo "$service is ready!"
            return 0
        fi
        
        echo "  $service not ready yet... ($count/$timeout)"
        sleep $interval
        count=$((count + interval))
    done
    
    echo "ERROR: $service failed to start within ${timeout}s"
    return 1
}

# Wait for MySQL to be ready (socket or TCP)
wait_for_service "MySQL" "mysqladmin ping -h 127.0.0.1 --silent" 60 2

# Wait for nginx
wait_for_service "nginx" "curl -f http://localhost >/dev/null" 30 2

# Wait for Tomcat (may take longer to start)
wait_for_service "Tomcat" "curl -f http://localhost:8080/manager/text/list >/dev/null 2>&1 || curl -f http://localhost:8080 >/dev/null" 90 3

echo "All services are ready!"